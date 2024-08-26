import pandas as pd
import datetime
import requests
from fake_useragent import UserAgent
from lxml import etree
import threading
import csv

provinces_citys = [['台北', '/taibei/']]

def get_admArea(weather_path, province_city):
    headers = {
        'User-Agent': str(UserAgent().random),
        'Cookie': 'Hm_lvt_a3f2879f6b3620a363bec646b7a8bcdd=1612610822,1614072517; lastCountyId=58027; lastCountyTime=1614080467; lastCountyPinyin=xuzhou; lastCityId=58027; lastProvinceId=25; Hm_lpvt_a3f2879f6b3620a363bec646b7a8bcdd=1614080468'
    }

    tianqi_url = 'http://tianqi.com'  # 历史天气
    city_tianqi_15_url = tianqi_url + province_city[1] + '15'  # 构建未来15天天气url
    print(city_tianqi_15_url)

    try:
        response = requests.get(city_tianqi_15_url, headers=headers, timeout=4).text
    except:
        response = requests.get(city_tianqi_15_url, headers=headers, timeout=4).text

    data = etree.HTML(response)
    li_list = data.xpath('//ul[@class="weaul"]/li')
    weather_data = []

    for li in li_list:
        date = li.xpath('./a/div[1]/span[1]/text()')[0]
        weather = li.xpath('./a/div[3]/text()')[0]
        temp_str = li.xpath('./a/div[4]/span[1]/text()')[0] + "~" + li.xpath('./a/div[4]/span[2]/text()')[0] + "℃"
        
        # 分割温度字符串为最低温和最高温
        low_temp, high_temp = temp_str.replace('℃', '').split('~')

        weather_data.append({
            "城市": province_city[0],
            "日期": '2023-' + date,
            "天气": weather,
            "最低温度": low_temp ,
            "最高温度": high_temp 
        })

    df = pd.DataFrame(weather_data)
    df.to_csv(weather_path, mode='a', header=False, encoding='utf-8', index=False)

if __name__ == '__main__':
    now_day = datetime.datetime.now().strftime("%Y-%m-%d")
    path_15_weather = now_day + 'citys_15_weather1.csv'
    header = ["城市", "日期", "天气", "最低温度", "最高温度"]

    # 写入头部信息
    with open(path_15_weather, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for citys in provinces_citys:
        t1 = threading.Thread(target=get_admArea, args=(path_15_weather, citys))
        t1.start()
