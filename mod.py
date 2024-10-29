import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

df=pd.read_csv("dataset_phishing.csv")
df=df.drop(labels='url', axis=1)
object_features=[col for col in df.columns if df[col].dtype=="O"]
class_labels=df['status'].unique().tolist()
class_labels.sort()
class_dict={}
for idx, label in enumerate(class_labels):
    class_dict[label]=idx
df['status']=df['status'].map(class_dict)

X=df.iloc[:,:-1]
y=df.iloc[:,-1:]
scaler=MinMaxScaler()
scaler.fit(X.values)
X_scaled=scaler.transform(X.values)
with open(file="scaler.pkl",mode="wb") as file:
    torch.save(scaler, file)

new_X=pd.DataFrame(data=X_scaled, columns=X.columns)
X_train, X_test, y_train, y_test=tts(new_X, y, test_size=0.2, random_state=42, 
                                     shuffle= True, stratify=y)
train_input_tensor=torch.from_numpy(X_train.values).float()
train_label_tensor=torch.from_numpy(y_train['status'].values).float()
val_input_tensor=torch.from_numpy(X_test.values).float()
val_label_tensor=torch.from_numpy(y_test['status'].values).float()

train_label_tensor=train_label_tensor.unsqueeze(1)
val_label_tensor=val_label_tensor.unsqueeze(1)

train_dataset=TensorDataset(train_input_tensor, train_label_tensor)
val_dataset=TensorDataset(val_input_tensor, val_label_tensor)

train_loader=DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader=DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self,dropout=0.4):
        super(MLP,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(in_features=55,out_features=300), 
            nn.ReLU(),
            nn.BatchNorm1d(num_features=300),
            nn.Dropout(p=dropout),
            
            nn.Linear(in_features=300,out_features=100),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=100),
            
            nn.Linear(in_features=100,out_features=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.network(x)
        return x

model=MLP(dropout=0.4)
optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001)
criterion=nn.BCELoss()
def train_loop(model,train_loader,val_loader,device,optimizer,criterion,batch_size,epochs):
    model=model.to(device)
    train_batch_size=len(train_loader)
    val_batch_size=len(val_loader)
    
    history={"train_accuracy":[],"train_loss":[],"val_accuracy":[],"val_loss":[]}
    
    for epoch in range(epochs):
        model.train() # training mode
        
        train_accuracy=0
        train_loss=0
        val_accuracy=0
        val_loss=0
        
        for X,y in train_loader:
            X=X.to(device)
            y=y.to(device)
            
            # forward propagation
            outputs=model(X)
            pred=torch.round(outputs)
            
            # loss computation
            loss=criterion(outputs,y)
            
            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cur_train_loss=loss.item()
            cur_train_accuracy=(pred==y).sum().item()/batch_size
            
            train_accuracy+=cur_train_accuracy
            train_loss+=cur_train_loss
        model.eval()
        with torch.no_grad():
            for X,y in val_loader:
                X=X.to(device)
                y=y.to(device)
                
                outputs=model(X)
                pred=torch.round(outputs)
                
                loss=criterion(outputs,y)
                
                cur_val_loss=loss.item()
                cur_val_accuracy=(pred==y).sum().item()/batch_size
                
                val_accuracy+=cur_val_accuracy
                val_loss+=cur_val_loss
        train_accuracy=train_accuracy/train_batch_size
        train_loss=train_loss/train_batch_size
        val_accuracy=val_accuracy/val_batch_size
        val_loss=val_loss/val_batch_size
        
        print(f"[{epoch+1:>3d}/{epochs:>3d}], train_accuracy:{train_accuracy:>5f}, train_loss:{train_loss:>5f}, val_accuracy:{val_accuracy:>5f}, val_loss:{val_loss:>5f}")
        history['train_accuracy'].append(train_accuracy)
        history['train_loss'].append(train_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_loss'].append(val_loss)

history=train_loop(model,train_loader,val_loader,device,optimizer,criterion,batch_size=32,epochs=100)

with open("phishing_model.pkl", "wb") as model_file:
    torch.save(model.state_dict(), model_file)
