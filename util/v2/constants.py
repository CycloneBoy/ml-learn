#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : constants.py
# @Author: sl
# @Date  : 2022/2/18 - 下午10:54
import os


class Constants:
    WORK_DIR = "/home/sl/workspace/python/a2020/ml-learn"

    LOG_DIR = os.path.join(WORK_DIR, "data/log")

    CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    DATA_MNIST_DIR = '~/workspace/data/mnist'
    MODEL_MNIST_DIR = '{}/models'.format(DATA_MNIST_DIR)

    DATA_FASHION_MNIST_DIR = '~/workspace/data/fashionmnist'

    DATA_DIR = '/home/sl/workspace/data'

    DATA_CACHE_DIR = '/home/sl/workspace/data/nlp'

    GLOVE_DATA_DIR = '/home/sl/workspace/data/nlp/glove/glove.6B'

    # aclImdb_v1.tar.gz
    IMDB_DATA_DIR = '/home/sl/workspace/data/nlp/aclImdb'

    # NLP模型保存
    MODEL_NLP_DIR = '/home/sl/workspace/data/nlp/model'

    # opencv 图片地址
    OPENCV_IMAGE_DIR = '/home/sl/data'
    # BILIBILI_VIDEO_IMAGE_DIR = '/home/sl/data/bilibili'
    BILIBILI_VIDEO_IMAGE_DIR = '/media/sl/Windows/share'

    ############################################################################
    # 文本相关的数据路径

    # 新闻语料库
    DATA_TXT_NEWS_DIR = '/home/sl/workspace/python/github/learning-nlp-master/chapter-3/data/news'
    # 停用词路径 合并github上的5个文件后的,停用词大小: 2524
    DATA_TXT_STOP_WORDS_DIR = '/home/sl/workspace/data/nlp/stopwords/stop_words.utf8'
    # github 上的停用词
    DATA_TXT_STOP_WORDS_GITHUB_DIR = '/home/sl/workspace/data/nlp/stopwords'

    # THUCNews 路径
    DATA_THUCNEWS_DIR = '/home/sl/workspace/python/a2020/ml-learn/data/nlp/THUCNews'

    # sgns.sogou.char  词嵌入
    DATA_EMBEDDING_SOGOU_CHAR = "/home/sl/workspace/data/nlp/sgns.sogou.char"
    # sgns.financial.bigram-char
    DATA_EMBEDDING_FINANCIAL_BIGRAM_CHAR = f"{DATA_CACHE_DIR}/sgns.financial.bigram-char"
    # DATA_EMBEDDING_FINANCIAL_BIGRAM_CHAR = f"{DATA_CACHE_DIR}/sgns.financial.bigram-char.bz2"

    NLP_PRETRAIN_DIR = DATA_CACHE_DIR
    BERT_BASE_CHINESE_TORCH = f'{DATA_CACHE_DIR}/bert-base-chinese'
    ############################################################################

    # 爬虫html页面
    DATA_HTML_DIR = os.path.join(WORK_DIR, "data/txt/html")

    # 爬虫问题保存地址
    DATA_QUESTION_DIR = os.path.join(WORK_DIR, "data/txt/result")

    # 爬虫问题保存地址 JSON
    DATA_JSON_DIR = os.path.join(WORK_DIR, "data/txt/json")
    DIR_DATA_JSON_CITY_INFO = os.path.join(WORK_DIR, "data/txt/json/city_info")

    # 问答
    DATA_QUESTION_ANSWER_DIR = os.path.join(DATA_CACHE_DIR, "question/result")

    QA_DELIMITER = "|"

    MAFENGWO_TOKEN_FILE = f"{WORK_DIR}/basis/spider/mafengwo/mafengwouser.token"
    MAFENGWO_COOKIE_FILE = f"{WORK_DIR}/basis/spider/mafengwo/mafengwoCookies.txt"

    ############################################################################
    # 测试数据

    # 问答测数据
    TEST_QA_1 = "363|成都啥时候去最合适，年后去合适吗|成都是座四季分明的城市，要论景色的话，自然是春秋两季为皆，春可赏花，樱花、海棠，腊梅、油菜花之类的，春色无边，秋天则可观银杏，枫叶等，景色怡人，气候舒适，但无论什么时候去，有阳光的日子就更棒了，如果只奔着美食或美女的话，则是一年四季都可以去的，没必要刻意找网红店，往那些居民区的小巷子里走，哪家人多就进哪家，定然不会让你失望，据说可以连吃一个月不重样，反正我只吃了一星期，就深信无疑，肥肠冒菜，各种串串，蹄花汤，豆汤饭，手工凉粉，有时候一个菜名也能分好几种做法，这是座来了就不想走的城市，一点也不夸张。|2018-10-12 18:26"
    TEST_QA_2 = "485|4/5-4/8西雅图周边SanJuan四天三夜房车行，再找一到两个小伙伴一起|当地华人社区发帖。|2021-03-30 10:55"
    TEST_QA_3 = "1548|TRANSAVIA航空在巴黎奥利机场值机问题|其实你在官网买票时，是否需要自己onlinecheckin，最早提前多少天onlinecheckin都有详细说明的，一般不是廉航，现场checkin不要钱|2017-08-23 16:24"
    TEST_QA_4 = "1698|上海到深圳自驾游攻略|根据导航就可以了。|2019-05-02 20:08"
    TEST_QA_5 = "1631|泰国落地签签证问题咨询。|没有限制。|2019-05-19 22:10"
    TEST_QA_6 = "710|关于莫斯科转机圣彼得堡的时间问题|先和携程客服确认下是不是联程，联程就不用自提行李了，他们网页上有的标的不准。如果需要取行李，那么就必须按照你了解到的流程走，2小时时间可能非常紧张，不一定够。|2018-01-04 23:33"
    TEST_QA_7 = "4535368|我是浙江护照，常驻日本，因为这次疫情会影响到我去别的国家出差吗|政策是随着疫情变化的，只能走一步看一步关注国家移民局公众号，可以查询最新的各国措施目前美国的措施是14天内不到访中国就可以入境但日本，早上看消息说，已经开始讨论对到过浙江的外国人采取措施，具体还要等他们拍板他们最近因为游轮也是焦头烂额，所以有可能措施越来越严|2020-02-1212:28"
    TEST_QA_8 = "1787|关于德国到瑞士的退税问题，求解答！|日内瓦机场二楼，有法国海关办公室，在日内瓦离境的时候可以去那里盖欧盟退税章。|2018-07-19 11:06"
    TEST_QA_9 = "646|巴黎进、巴塞罗那出！西班牙行程安排提问|第一，就我所知，塞维利亚是在马德里的下面，但这不是重点。西班牙行程可以这样安排，巴黎飞机前往塞维利亚，然后塞维利亚火车前往马德里，最后马德里火车或者飞机前往巴塞罗那。这是合理的。第二，不知道楼主的整体旅游时间是多少，但个人建议，塞维利亚两天，马德里三天，巴塞罗那四天。个人对塞维利亚有所偏爱，甚至愿意添加一天。楼主可以根据自己的喜爱进行调整。|2017-11-13 10:47"
    TEST_QA_10 = "866|求助，圣诞节的时候国内哪里好玩啊？|哈尔滨当然冷。。。。不过圣诞气氛够强的，我想都是香港或者外国吧我上年去了新加坡，挺好的，不算很热。。。整条乌节路都是各式各样的圣诞树哦。。。还有很多美食。。|2011-10-12 04:58"

    TEST_QA_LIST_10 = [TEST_QA_1, TEST_QA_2, TEST_QA_3, TEST_QA_4, TEST_QA_5, TEST_QA_6, TEST_QA_7, TEST_QA_8,
                       TEST_QA_9,
                       TEST_QA_10]

    # ML
    ML_DATA_DIR = os.path.join(WORK_DIR, "data/ml")

    SEEDS_DATA_DIR = os.path.join(ML_DATA_DIR, 'seeds_dataset.txt')
    CONCRETE_DATA_DIR = os.path.join(ML_DATA_DIR, 'Concrete_Data.csv')

    # 摘要测试数据
    TEST_SUMMARY_1 = ("方直科技拟以自有资金1.2亿元共同投资设立嘉道方直投资基金",
                      "多知网5月26日消息，今日，方直科技发公告，拟用自有资金人民币1.2亿元，与深圳嘉道谷投资管理有限公司、深圳嘉道功程股权投资基金（有限合伙）共同发起设立嘉道方直教育产业投资基金（暂定名）。该基金认缴出资总规模为人民币3.01亿元。基金的出资方式具体如下：![\\](http://www.duozhi.com/uploadfile/2017/0526/20170526092512491.png)出资进度方面，基金合伙人的出资应于基金成立之日起四年内分四期缴足，每期缴付7525万元；各基金合伙人每期按其出资比例缴付。合伙期限为11年，投资目标为教育领域初创期或成长期企业。截止公告披露日，深圳嘉道谷投资管理有限公司股权结构如下:![\\](http://www.duozhi.com/uploadfile/2017/0526/20170526092525724.png)截止公告披露日，深圳嘉道功程股权投资基金产权结构如下:![\\](http://www.duozhi.com/uploadfile/2017/0526/20170526092538802.png)公告还披露，方直科技将探索在中小学教育、在线教育、非学历教育、学前教育、留学咨询等教育行业其他分支领域的投资。方直科技2016年营业收入9691万元，营业利润1432万元，归属于普通股股东的净利润1847万元。（多知网 黎珊）")

    TEST_SUMMARY_DOC = "是上世纪90年代末提出的一种计算网页权重的算法。" \
                       "当时，互联网技术突飞猛进，各种网页网站爆炸式增长，" \
                       "业界急需一种相对比较准确的网页重要性计算方法，" \
                       "是人们能够从海量互联网世界中找出自己需要的信息。" \
                       "百度百科如是介绍他的思想:PageRank通过网络浩瀚的超链接关系来确定一个页面的等级。" \
                       "Google把从A页面到B页面的链接解释为A页面给B页面投票，" \
                       "Google根据投票来源甚至来源的来源，即链接到A页面的页面" \
                       "和投票目标的等级来决定新的等级。简单的说，" \
                       "一个高等级的页面可以使其他低等级页面的等级提升。" \
                       "PageRank The PageRank Citation Ranking: Bringing Order to the Web，" \
                       "具体说来就是，PageRank有两个基本思想，也可以说是假设，" \
                       "即数量假设：一个网页被越多的其他页面链接，就越重）；" \
                       "质量假设：一个网页越是被高质量的网页链接，就越重要。" \
                       "总的来说就是一句话，从全局角度考虑，获取重要的信息。"

    # 多标签
    QA_DATA_DIR = f"{DATA_CACHE_DIR}/multi_class"
    QA_ALL_SORT_DATA_DIR = f"{QA_DATA_DIR}/qa_data_all_no_label_sort.csv"
    QA_ALL_SORT_CUT_DATA_DIR = f"{QA_DATA_DIR}/qa_data_all_cut.txt"
    QA_ALL_SORT_MERGE_CUT_DATA_DIR = f"{QA_DATA_DIR}/qa_data_all_merge_cut.csv"
    QA_QUESTION_DATA_DIR = f"{QA_DATA_DIR}/qa_question.csv"
    QA_TYPE_DATA_DIR = f"{QA_DATA_DIR}/qatype.txt"

    # 熵简科技 AI Lab 近期开源了基于 BERT 架构的金融领域预训练语言模型 FinBERT 1.0
    MODEL_NLP_BERT_FINBERT_TF = f"{DATA_CACHE_DIR}/FinBERT_L-12_H-768_A-12_tf"
    # bert-base-chinese
    MODEL_NLP_BERT_CHINESE_TF = f"{DATA_CACHE_DIR}/chinese_L-12_H-768_A-12"
    RUN_MODEL = MODEL_NLP_BERT_FINBERT_TF

    # Chinese Medical Intent Dataset(CMID)
    DATA_NLP_CLASS_CMID = f"{DATA_CACHE_DIR}/CMID.json"
    DATA_NLP_CLASS_DISEASES = f"{DATA_CACHE_DIR}/diseases.json"

    ############################################################################
    # mafengwo

    SPIDER_DIR = "/home/sl/workspace/data/spider"
    SPIDER_MAFENGWO_DIR = "/home/sl/workspace/data/spider/mafengwo"
    SPIDER_MAFENGWO_CITY_ALL_DIR = "/home/sl/workspace/data/spider/mafengwo/www.mafengwo.cn_mdd_.html"
    SPIDER_MFW_CITY_YUNNAN_DIR = "/home/sl/workspace/data/spider/mafengwo/travel-scenic-spot/mafengwo/12711.html"
    SPIDER_MFW_CITY_INFO_DIR = f"{SPIDER_MAFENGWO_DIR}/city_info"

    #
    VIDEO_EYE_MOTION_DIR = f"{DATA_DIR}/video/eye_motion.mp4"
