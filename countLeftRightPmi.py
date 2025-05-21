import re
import math
from collections import Counter

target_words = ('special edition', 'limited edition')
offset = 12
freq_one = 5 + offset * 2
freq_two = 5 + offset
freq_left_right = 5 + offset
pmi_limit = 1.5

file_path = r'C:\....txt'

SPAN = 5  # 左右各5个词的范围

def calculate_pmi(ab_freq, a_freq, b_freq, corpus_size, span):
    """
    计算PMI值
    ab_freq: 目标短语和搭配词共现频率
    a_freq: 目标短语频率
    b_freq: 搭配词频率
    corpus_size: 语料库大小（总词数）
    span: 窗口大小
    """
    try:
        mi = math.log((ab_freq * corpus_size) / (a_freq * b_freq * span)) / math.log(2) # span * 2 后，pmi结果-1
        return mi
    except (ValueError, ZeroDivisionError):
        return float('-inf')

def count_corpus_size(sentences):
    """计算语料库总词数"""
    return sum(len(sentence.split()) for sentence in sentences)

def find_neighbors_and_count(grouped_rows, target_phrase):
    # 初始化计数器
    target_freq = 0  # 目标短语的总频率
    collocate_freq = Counter()  # 所有单词的频率
    left_cooccurrence_freq = Counter()  # 左侧共现频率
    right_cooccurrence_freq = Counter()  # 右侧共现频率
    left_bigrams_counter = Counter()  # 左侧bigram频率
    right_bigrams_counter = Counter()  # 右侧bigram频率
    left_right_pair_counter = Counter()  # 左右词对的共现频率
    
    # 计算语料库大小
    corpus_size = count_corpus_size(grouped_rows)
    
    # 遍历每个句子
    for sentence in grouped_rows:
        words = sentence.split()
        
        # 更新单词总频率
        collocate_freq.update(words)
        
        # 查找目标短语
        pattern = r'\b' + re.escape(target_phrase) + r'\b'
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE): # 忽略大小写
            target_freq += 1
            target_index = len(sentence[:match.start()].split())
            
            # 获取左右各5个词的范围
            left_words = words[max(0, target_index - SPAN):target_index]
            right_words = words[target_index + len(target_phrase.split()):
                              target_index + len(target_phrase.split()) + SPAN]
            
            # 分别更新左右共现频率
            left_cooccurrence_freq.update(left_words)
            right_cooccurrence_freq.update(right_words)
            
            # 统计左右词对的共现
            for left_word in left_words:
                for right_word in right_words:
                    if left_word.lower() not in coca_words or right_word.lower() not in coca_words: # 20250321_163705 and 改成 or
                        pair = f"{left_word}@@@{right_word}"
                        left_right_pair_counter[pair] += 1
            
            # 统计左侧bigrams
            for i in range(len(left_words) - 1):
                bigram = f"{left_words[i]} {left_words[i + 1]}"
                left_bigrams_counter[bigram] += 1
                
            # 统计右侧bigrams
            for i in range(len(right_words) - 1):
                bigram = f"{right_words[i]} {right_words[i + 1]}"
                right_bigrams_counter[bigram] += 1
    
    # 计算左右词对的PMI
    left_right_pair_pmi = {}
    for pair, co_freq in left_right_pair_counter.items():
        left_word, right_word = pair.split('@@@')
        # 使用左右词各自的频率作为分母
        pmi = calculate_pmi(co_freq, 1, 
                          collocate_freq[left_word] * collocate_freq[right_word], 
                          corpus_size, SPAN) # / corpus_size 不能去掉不然都变成-数了  #这样好像好一些，下面这计算方式让pmi太大10左右
        left_right_pair_pmi[pair] = (pmi, co_freq)
    
    # 分别计算左右PMI值
    left_pmi_scores = {}
    right_pmi_scores = {}
    
    # 计算左侧PMI
    for word, co_freq in left_cooccurrence_freq.items():
        if word.lower() not in coca_words:  # 过滤停用词
            pmi = calculate_pmi(co_freq, target_freq, collocate_freq[word], corpus_size, SPAN) 
            left_pmi_scores[word] = (pmi, co_freq)
    
    # 计算右侧PMI
    for word, co_freq in right_cooccurrence_freq.items():
        if word.lower() not in coca_words:  # 过滤停用词
            pmi = calculate_pmi(co_freq, target_freq, collocate_freq[word], corpus_size, SPAN)
            right_pmi_scores[word] = (pmi, co_freq)
    
    # 处理bigrams
    left_bigrams_pmi = {}
    right_bigrams_pmi = {}
    
    # 计算所有bigrams的总频率
    all_bigrams_freq = Counter()
    for sentence in grouped_rows:
        words = sentence.split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            all_bigrams_freq[bigram] += 1
    
    # 计算左侧bigrams的PMI
    for bigram, co_freq in left_bigrams_counter.items():
        pmi = calculate_pmi(co_freq, target_freq, all_bigrams_freq[bigram], corpus_size, SPAN)
        left_bigrams_pmi[bigram] = (pmi, co_freq)

    # 计算右侧bigrams的PMI
    for bigram, co_freq in right_bigrams_counter.items():
        pmi = calculate_pmi(co_freq, target_freq, all_bigrams_freq[bigram], corpus_size, SPAN)
        right_bigrams_pmi[bigram] = (pmi, co_freq)

    # 按频率排序
    sorted_left_bigrams = sorted(left_bigrams_pmi.items(), key=lambda x: x[1][1], reverse=True)
    sorted_right_bigrams = sorted(right_bigrams_pmi.items(), key=lambda x: x[1][1], reverse=True)

    # 按频率排序左右词对
    sorted_left_right_pairs = sorted(left_right_pair_pmi.items(), key=lambda x: x[1][1], reverse=True)

    # 按频率排序
    sorted_left_collocations = sorted(left_pmi_scores.items(), key=lambda x: x[1][1], reverse=True) 
    sorted_right_collocations = sorted(right_pmi_scores.items(), key=lambda x: x[1][1], reverse=True) #[1][0]是pmi [1][1]是频率
    
    return (sorted_left_collocations, sorted_right_collocations, sorted_left_bigrams, sorted_right_bigrams, sorted_left_right_pairs), target_freq, corpus_size

def read_words_from_file():
    file_path = r'C:\...\cocaForms5000.txt'
    # 读取文件并将每一行添加到列表中
    with open(file_path, 'r', encoding='utf-8') as file:
        word_list = [line.strip() for line in file]

    return word_list

coca_words = read_words_from_file()[:40]
# 添加标点符号到停用词列表
coca_words.extend([',', '.', '’', '?', '“', ':', ')', '!', '”'])

# 从文件中读取句子
def read_sentences_from_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if "出现频率" in line:
                break
            sentences.append(line.strip())
    return sentences


# 从文件中读取句子
grouped_rows = read_sentences_from_file(file_path)

# 对每个目标短语计算PMI
for target_phrase in target_words:
    print(f"\n\n\n分析短语: '{target_phrase}'")
    (left_collocations, right_collocations, left_bigrams, right_bigrams, left_right_pairs), phrase_freq, corpus_size = find_neighbors_and_count(grouped_rows, target_phrase)
    
    print(f"目标短语'{target_phrase}'在语料库中出现{phrase_freq}次")
    print(f"语料库总词数: {corpus_size}")
    
    # 计算最长单词长度，用于对齐
    max_word_len = 10 # 生成的那个max长度太长了 (9-10) * " " 结果''
    max_bigram_len = 15
    
    print("\n左侧最强搭配（按频率排序）:" + f" {freq_one} {pmi_limit}")
    print("单词\tPMI值\t共现频率")
    # print("-" * 40)
    for word, (pmi, co_freq) in left_collocations:
        if pmi > pmi_limit and co_freq >= freq_one:
            spaces = " " * (max_word_len - len(word))
            print(f"{word}{spaces}{pmi:.2f}\t{co_freq}")
            
    print("\n左侧最常见的两词组合及其频率:" + f" {freq_two}")
    print("词组\tPMI值\t频率")
    # print("-" * 40)
    for bigram, (pmi, freq) in left_bigrams:
        if pmi > pmi_limit and freq >= freq_two:
            spaces = " " * (max_bigram_len - len(bigram))
            print(f"{bigram}{spaces}{pmi:.2f}\t{freq}")
            
    print("\n右侧最强搭配（按频率排序）:")
    print("单词\tPMI值\t共现频率")
    # print("-" * 40)
    for word, (pmi, co_freq) in right_collocations:
        if pmi > pmi_limit and co_freq >= freq_one:
            spaces = " " * (max_word_len - len(word))
            print(f"{word}{spaces}{pmi:.2f}\t{co_freq}")
            
    print("\n右侧最常见的两词组合及其频率:")
    print("词组\tPMI值\t频率")
    # print("-" * 40)
    for bigram, (pmi, freq) in right_bigrams:
        if pmi > pmi_limit and freq >= freq_two:
            spaces = " " * (max_bigram_len - len(bigram))
            print(f"{bigram}{spaces}{pmi:.2f}\t{freq}")
            
    print("\n左右词对的搭配（按频率排序）: " + str(freq_left_right))
    print("左词-右词" + " " * (max_bigram_len - 6) + "\tPMI值\t共现频率")
    # print("-" * 40)
    for pair, (pmi, freq) in left_right_pairs:
        if pmi > pmi_limit and freq >= freq_left_right:
            left_word, right_word = pair.split('@@@')
            pair_display = f"{left_word}-{right_word}"
            spaces = " " * (max_bigram_len - len(pair_display))
            print(f"{pair_display}{spaces}{pmi:.2f}\t{freq}")
