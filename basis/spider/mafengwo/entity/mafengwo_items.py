# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class MafengwoRedisItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass

# 蚂蜂窝游记
class MafengwoTravelNotes(scrapy.Item):
    id = scrapy.Field()            # 游记id
    url = scrapy.Field()            # 游记链接
    destination = scrapy.Field()    # 游记目的地
    author_url = scrapy.Field()     # 游记作者首页链接
    author_name = scrapy.Field()    # 游记作者名称
    image_url = scrapy.Field()      # 游记封面图片链接
    year = scrapy.Field()           # 游记年份


# 蚂蜂窝热门目的地
class MafengwoTravelDestination(scrapy.Item):
    id = scrapy.Field()  # 游记ID
    country_name = scrapy.Field()            # 热门目的地国家名称
    country_url = scrapy.Field()             # 热门目的地国家链接
    city_name = scrapy.Field()               # 热门目的地城市名称
    city_name_en = scrapy.Field()              # 热门目的地城市英文名称
    city_url = scrapy.Field()                # 热门目的地城市链接
    city_index = scrapy.Field()               # 热门目的地城市的编码
    crawl_status = scrapy.Field()            # 是否已经爬取
    crawl_time = scrapy.Field()              # 爬取时间
    total_page = scrapy.Field()              # 游记总页数
    total_number = scrapy.Field()              # 游记总数
    image_total_number = scrapy.Field()        # 游记图片总数


# 蚂蜂窝游记
class MafengwoTravel(scrapy.Item):
    travel_url = scrapy.Field()             # 游记链接
    travel_name = scrapy.Field()            # 游记名称
    travel_type = scrapy.Field()            # 游记分类 ： 宝藏 、 星级
    travel_summary = scrapy.Field()         # 游记摘要
    travel_destination = scrapy.Field()     # 游记目的地
    travel_destination_country = scrapy.Field()     # 游记目的地国家
    travel_image_url = scrapy.Field()       # 游记封面图片链接
    author_id = scrapy.Field()              # 游记作者ID
    author_url = scrapy.Field()             # 游记作者首页链接
    author_name = scrapy.Field()            # 游记作者名称
    author_image_url = scrapy.Field()       # 游记作者图像链接
    travel_view_count = scrapy.Field()      # 游记浏览总数
    travel_comment_count = scrapy.Field()   # 游记评论总数
    travel_up_count = scrapy.Field()        # 游记顶的总数
    crawl_status = scrapy.Field()           # 是否已经爬取
    crawl_time = scrapy.Field()             # 爬取时间
    travel_father_id = scrapy.Field()       # 游记父亲id
    travel_id = scrapy.Field()              # 游记ID


# 蚂蜂窝热门目的地
class XiciProxyItem(scrapy.Item):
    ip = scrapy.Field()            # IP
    port = scrapy.Field()             # port
    scheme = scrapy.Field()               # http /https
    speed = scrapy.Field()                # 速度
    survival_time = scrapy.Field()               # 存活時間
    ip_port = scrapy.Field()               # 存活時間
    proxy = scrapy.Field()               # 存活時間


# 蚂蜂窝游记
class MafengwoTravelDetail(scrapy.Item):
    id = scrapy.Field()             # 游记链接ID
    travel_url = scrapy.Field()             # 游记链接
    travel_name = scrapy.Field()            # 游记名称
    travel_type = scrapy.Field()            # 游记分类 ： 宝藏 、 星级
    travel_summary = scrapy.Field()         # 游记摘要
    travel_destination = scrapy.Field()     # 游记目的地
    travel_destination_country = scrapy.Field()     # 游记目的地国家
    travel_image_url = scrapy.Field()       # 游记封面图片链接
    author_id = scrapy.Field()              # 游记作者ID
    author_url = scrapy.Field()             # 游记作者首页链接
    author_name = scrapy.Field()            # 游记作者名称
    author_image_url = scrapy.Field()       # 游记作者图像链接
    travel_view_count = scrapy.Field()      # 游记浏览总数
    travel_comment_count = scrapy.Field()   # 游记评论总数
    travel_up_count = scrapy.Field()        # 游记顶的总数
    crawl_status = scrapy.Field()           # 是否已经爬取
    crawl_time = scrapy.Field()             # 爬取时间
    travel_father_id = scrapy.Field()       # 游记父亲id
    travel_id = scrapy.Field()              # 游记ID

    travel_home_image_url = scrapy.Field()  # 是否已经爬取
    travel_summary_all = scrapy.Field()             # 爬取时间
    travel_share_count = scrapy.Field()       # 游记父亲id
    travel_collect_count = scrapy.Field()              # 游记ID
    travel_time = scrapy.Field()              # 游记ID
    travel_day = scrapy.Field()              # 游记ID
    travel_people = scrapy.Field()              # 游记ID
    travel_cost = scrapy.Field()              # 游记ID

    travel_country_name = scrapy.Field()              # 游记ID
    travel_country_url = scrapy.Field()              # 游记ID
    travel_country_image_url = scrapy.Field()              # 游记ID
    travel_travel_country_image_countcost = scrapy.Field()              # 游记ID

    travel_word_count = scrapy.Field()              # 游记ID
    travel_image_count = scrapy.Field()              # 游记ID
    travel_help_person_count = scrapy.Field()              # 游记ID
    travel_first_image_list_url = scrapy.Field()              # 游记ID