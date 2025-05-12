import pandas as pd
import os
import glob
from collections import Counter
import statistics
# DGA（Domain Generation Algorithm）域名特征提取是一种用于检测恶意软件和网络攻击的技术。下面是一些常见的DGA域名特征提取的方向：

# 域名长度特征：DGA生成的域名通常具有特定的长度范围，可以通过统计域名长度分布来进行特征提取。

# 域名字符集特征：DGA生成的域名通常使用特定的字符集，如只包含字母、数字或特定的字符组合。通过分析域名中包含的字符类型和字符集分布，可以提取特征。

# 域名频率特征：DGA生成的域名通常具有较高的频率，即生成的域名在短时间内大量出现。可以通过统计域名的出现频率来进行特征提取。

# 域名结构特征：DGA生成的域名通常具有特定的结构或模式，如特定的前缀、后缀、分隔符等。可以通过分析域名的结构来提取特征。

# 域名时间特征：DGA生成的域名通常具有特定的时间相关特征，如在特定的时间间隔内生成的域名具有相似的特征。可以通过分析域名生成的时间模式来提取特征。

# 域名语义特征：DGA生成的域名通常缺乏语义意义，不符合正常域名的命名规则。可以通过分析域名的语义特征，如是否包含常见的单词、词典中的词等，来进行特征提取。

# 域名排列组合特征：DGA生成的域名通常具有排列组合的特征，即使用特定的字符组合、组件或模式生成域名。可以通过分析域名的排列组合特征来进行特征提取。

# 域名网络行为特征：DGA生成的域名通常与恶意软件的网络行为相关联，如与C&C（Command and Control）服务器进行通信等。可以通过分析域名与其他网络行为的关联来提取特征。
from collections import Counter
import statistics
from collections import Counter
import statistics

class DGADomainAnalyzer:
    def __init__(self, domains):
        self.domains = domains

    def calculate_repeated_characters(self):
        repeated_characters = []
        for domain in self.domains:
            if len(domain) > 0:
                repeated_characters.append(max(Counter(domain).values()))  # 计算重复字符的最大出现次数
            else:
                repeated_characters.append(0)  # 如果域名为空字符串，则将重复字符的最大出现次数设置为0
        return repeated_characters

    def calculate_character_frequency(self):
        character_frequencies = []
        for domain in self.domains:
            if len(domain) > 0:
                character_frequencies.append(Counter(domain))  # 计算字符频率
            else:
                character_frequencies.append(Counter())  # 如果域名为空字符串，则将字符频率设置为空Counter对象
        return character_frequencies

    def calculate_vowel_consonant_ratio(self):
        vowel_consonant_ratios = []
        for domain in self.domains:
            if len(domain.replace(' ', '')) > 0:
                vowel_count = domain.count('a') + domain.count('e') + domain.count('i') + domain.count('o') + domain.count('u')  # 计算元音字母的数量
                consonant_count = len(domain) - domain.count(' ') - vowel_count  # 计算辅音字母的数量
                vowel_consonant_ratios.append(vowel_count / consonant_count)  # 计算元音字母与辅音字母的比例
            else:
                vowel_consonant_ratios.append(0)  # 如果域名为空字符串或只包含空格，则将元音辅音比例设置为0
        return vowel_consonant_ratios

    def calculate_letter_digit_ratio(self):
        letter_digit_ratios = []
        for domain in self.domains:
            if len(domain.replace(' ', '')) > 0 and domain.count('0') + domain.count('1') + domain.count('2') + domain.count('3') + domain.count('4') + domain.count('5') + domain.count('6') + domain.count('7') + domain.count('8') + domain.count('9') > 0:
                letter_count = sum(c.isalpha() for c in domain)  # 计算字母的数量
                digit_count = sum(c.isdigit() for c in domain)  # 计算数字的数量
                letter_digit_ratios.append(letter_count / digit_count)  # 计算字母与数字的比例
            else:
                letter_digit_ratios.append(0)  # 如果域名为空字符串、只包含空格或不包含任何数字，则将字母数字比例设置为0
        return letter_digit_ratios

    def calculate_domain_length(self):
        domain_lengths = [len(domain) for domain in self.domains]  # 计算域名的长度
        return domain_lengths

    def calculate_max_min_avg_median_mode(self, values):
        if len(values) > 0:
            return max(values), min(values), sum(values) / len(values), statistics.median(values), statistics.mode(values)  # 计算值的最大值、最小值、平均值、中位数和众数
        else:
            return 0, 0, 0, 0, None  # 如果值为空列表，则将所有统计结果设置为0

    def analyze_domains(self):
        repeated_characters = self.calculate_repeated_characters()
        repeated_characters_max, repeated_characters_min, repeated_characters_avg, repeated_characters_median, repeated_characters_mode = self.calculate_max_min_avg_median_mode(repeated_characters)

        character_frequencies = self.calculate_character_frequency()

        vowel_consonant_ratios = self.calculate_vowel_consonant_ratio()
        vowel_consonant_ratio_max, vowel_consonant_ratio_min, vowel_consonant_ratio_avg, vowel_consonant_ratio_median, vowel_consonant_ratio_mode = self.calculate_max_min_avg_median_mode(vowel_consonant_ratios)

        letter_digit_ratios = self.calculate_letter_digit_ratio()
        letter_digit_ratio_max, letter_digit_ratio_min, letter_digit_ratio_avg, letter_digit_ratio_median, letter_digit_ratio_mode = self.calculate_max_min_avg_median_mode(letter_digit_ratios)

        domain_lengths = self.calculate_domain_length()
        domain_length_max, domain_length_min, domain_length_avg, domain_length_median, domain_length_mode = self.calculate_max_min_avg_median_mode(domain_lengths)

        analysis_result = {
            "Repeated Characters": {
                # "values": repeated_characters,
                "max": repeated_characters_max,
                "min": repeated_characters_min,
                "avg": repeated_characters_avg,
                "median": repeated_characters_median,
                "mode": repeated_characters_mode
            },
            # "Character Frequencies": character_frequencies,
            "Vowel-Consonant Ratios": {
                # "values": vowel_consonant_ratios,
                "max": vowel_consonant_ratio_max,
                "min": vowel_consonant_ratio_min,
                "avg": vowel_consonant_ratio_avg,
                "median": vowel_consonant_ratio_median,
                "mode": vowel_consonant_ratio_mode
            },
            "Letter-Digit Ratios": {
                # "values": letter_digit_ratios,
                "max": letter_digit_ratio_max,
                "min": letter_digit_ratio_min,
                "avg": letter_digit_ratio_avg,
                "median": letter_digit_ratio_median,
                "mode": letter_digit_ratio_mode
            },
            "Domain Lengths": {
                # "values": domain_lengths,
                "max": domain_length_max,
                "min": domain_length_min,
                "avg": domain_length_avg,
                "median": domain_length_median,
                "mode": domain_length_mode
            }
        }

        return analysis_result
    
    
# 工具函数，汇聚文件读入，特征处理的工作
def diff_file_comparation(Analyzer):
    benign_folder_path = 'data/Benign'  # 文件夹的路径
    malicious_folder_path = 'data/DGA/2016-09-19-dgarchive_full'
    
    # 良性文件路径
    benign_file_paths = glob.glob(os.path.join(benign_folder_path, '*'))
    
    # 恶性文件路径
    malicious_file_paths = glob.glob(os.path.join(malicious_folder_path, '*'))
    
    # 记录良性域名的数据
    benign_data_calculate = []
    malicious_data_calculate = []
    # 良性域名统计
    for file in benign_file_paths:
        # 读文件
        data = pd.read_csv(file)
        # 取出域名
        domains = data['domain'].unique()
        # 进行统计，添加结果
        benign_data_calculate.append(['benign',(Analyzer(domains)).analyze_domains()]) 
    
    # 恶意域名统计
    for file in malicious_file_paths:
        # 读文件
        data = pd.read_csv(file)
        # 拉出恶意域名
        domains = data.iloc[:,0].tolist()
        # 获取DGA家族
        label = data.iloc[:,0].to_list()[0]
        #获取结果
        malicious_data_calculate.append([label,(Analyzer(domains)).analyze_domains()]) 
    return benign_data_calculate,malicious_data_calculate

print(diff_file_comparation(DGADomainAnalyzer))
    
    