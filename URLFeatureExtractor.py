import urllib.parse

class URLFeatureExtractor:
    def __init__(self, url):
        self.url = url
        self.parsed_url = urllib.parse.urlparse(url)

    def get_length_url(self):
        return len(self.url)

    def get_length_hostname(self):
        return len(self.parsed_url.hostname) if self.parsed_url.hostname else 0

    def get_ip(self):
        return 1 if self.parsed_url.hostname and self.parsed_url.hostname.replace(".", "").isdigit() else 0

    def get_nb_dots(self):
        return self.url.count(".")

    def get_nb_hyphens(self):
        return self.url.count("-")

    def get_nb_at(self):
        return self.url.count("@")

    def get_nb_qm(self):
        return self.url.count("?")

    def get_nb_and(self):
        return self.url.count("&")

    def get_nb_or(self):
        return self.url.count("|")

    def get_nb_eq(self):
        return self.url.count("=")

    def get_nb_underscore(self):
        return self.url.count("_")

    def get_nb_tilde(self):
        return self.url.count("~")

    def get_nb_percent(self):
        return self.url.count("%")

    def get_nb_slash(self):
        return self.url.count("/")

    def get_nb_star(self):
        return self.url.count("*")

    def get_nb_colon(self):
        return self.url.count(":")

    def get_nb_comma(self):
        return self.url.count(",")

    def get_nb_semicolon(self):
        return self.url.count(";")

    def get_nb_dollar(self):
        return self.url.count("$")

    def get_nb_space(self):
        return self.url.count(" ")

    def get_nb_www(self):
        return self.url.count("www.")

    def get_nb_com(self):
        return self.url.count(".com")

    def get_nb_dslash(self):
        return self.url.count("//")

    def get_http_in_path(self):
        return 1 if "http://" in self.url else 0

    def get_https_token(self):
        return 1 if "https" in self.url else 0

    def get_ratio_digits_url(self):
        return sum(c.isdigit() for c in self.url) / len(self.url)

    def get_ratio_digits_host(self):
        return sum(c.isdigit() for c in self.parsed_url.hostname) / len(self.parsed_url.hostname)

    def get_punycode(self):
        return 1 if self.parsed_url.hostname.encode("idna").decode("utf-8") != self.parsed_url.hostname else 0

    def get_port(self):
        return self.parsed_url.port if self.parsed_url.port else 0

    def get_tld_in_path(self):
        return 1 if self.parsed_url.path.endswith((".com", ".net", ".org")) else 0

    def get_tld_in_subdomain(self):
        return 1 if "." in self.parsed_url.hostname.split(".")[0] else 0

    def get_abnormal_subdomain(self):
        return 1 if sum(c.isalnum() or c == "-" for c in self.parsed_url.hostname) != len(self.parsed_url.hostname) else 0

    def get_nb_subdomains(self):
        return len(self.parsed_url.hostname.split(".")) - 2

    def get_prefix_suffix(self):
        return 1 if "-" in self.parsed_url.hostname.split(".")[0] or "-" in self.parsed_url.hostname.split(".")[-1] else 0

    def get_random_domain(self):
        return 1 if self.parsed_url.hostname.replace(".", "").isalnum() else 0

    def get_shortening_service(self):
        return 1 if self.parsed_url.hostname.endswith(("bit.ly", "t.co", "goo.gl")) else 0

    def get_path_extension(self):
        return 1 if "." in self.parsed_url.path.split("/")[-1] else 0

    def get_nb_redirection(self):
        return 1 if "//" in self.url else 0

    def get_nb_external_redirection(self):
        return 1 if "://" in self.url else 0

    def get_length_words_raw(self):
        return len(self.url.split("/"))

    def get_char_repeat(self):
        return max(self.url.count(char) for char in set(self.url))

    def get_shortest_words_raw(self):
        return min(len(word) for word in self.url.split("/"))

    def get_shortest_word_host(self):
        return min(len(word) for word in self.parsed_url.hostname.split("."))

    def get_shortest_word_path(self):
        return min(len(word) for word in self.parsed_url.path.split("/"))

    def get_longest_words_raw(self):
        return max(len(word) for word in self.url.split("/"))

    def get_longest_word_host(self):
        return max(len(word) for word in self.parsed_url.hostname.split("."))

    def get_longest_word_path(self):
        return max(len(word) for word in self.parsed_url.path.split("/"))

    def get_avg_words_raw(self):
        length_words_raw = self.get_length_words_raw()
        return sum(len(word) for word in self.url.split("/")) / length_words_raw if length_words_raw > 0 else 0

    def get_avg_word_host(self):
        parsed_hostname = self.parsed_url.hostname.split(".")
        return sum(len(word) for word in parsed_hostname) / (len(parsed_hostname) - 1) if len(parsed_hostname) > 1 else 0

    def get_avg_word_path(self):
        length_words_raw = self.get_length_words_raw()
        return sum(len(word) for word in self.parsed_url.path.split("/")) / length_words_raw if length_words_raw > 0 else 0

    def get_phish_hints(self):
        return 1 if "phish" in self.url else 0

    def get_domain_in_brand(self):
        return 1 if "brand" in self.url else 0

    def get_brand_in_subdomain(self):
        return 1 if "brand" in self.parsed_url.hostname.split(".") else 0

    def get_brand_in_path(self):
        return 1 if "brand" in self.parsed_url.path.split("/") else 0

    def get_suspecious_tld(self):
        return 1 if self.parsed_url.hostname.endswith((".xyz", ".top", ".pw")) else 0
    
    def get_all_features(self):
        feature_order = [
            "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at", "nb_qm", "nb_and",
            "nb_or", "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash", "nb_star", "nb_colon",
            "nb_comma", "nb_semicolon", "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash", "http_in_path",
            "https_token", "ratio_digits_url", "ratio_digits_host", "punycode", "port", "tld_in_path",
            "tld_in_subdomain", "abnormal_subdomain", "nb_subdomains", "prefix_suffix", "random_domain",
            "shortening_service", "path_extension", "nb_redirection", "nb_external_redirection", "length_words_raw",
            "char_repeat", "shortest_words_raw", "shortest_word_host", "shortest_word_path", "longest_words_raw",
            "longest_word_host", "longest_word_path", "avg_words_raw", "avg_word_host", "avg_word_path", "phish_hints",
            "domain_in_brand", "brand_in_subdomain", "brand_in_path", "suspecious_tld"
        ]
        feature_values = []
        for feature_name in feature_order:
            method_name = f"get_{feature_name}"
            method = getattr(self, method_name)
            feature_values.append(method())
        return [feature_values]
    
"""url='https://www.google.com/'
result = URLFeatureExtractor(url)
features=result.get_all_features()
print(features)"""