# API接口

## 1. 食品安全新闻判断接口
### 1.1 功能描述
根据新闻标题判断是不是食品安全新闻
### 1.2 请求说明
> 请求方式：POST<br>
请求URL ：['http:0.0.0.0:5000/news_recognition'](#)

### 1.3 请求参数
字段       |字段类型       |字段说明
------------|-----------|-----------
news_title       |string        |新闻标题

### 1.3.1 样例
```
{
  "news_title": "空军发布多语种宣传片战神绕岛新航迹含闽南语等",
}
```
### 1.4 返回结果
```json  
{
  "code": "0",
}
``` 
### 1.5 返回参数
字段       |字段类型       |字段说明
------------|-----------|-----------
code       |string        |状态码

### 1.6 状态码类型
状态码       |说明
------------|-----------
1       |是食品安全新闻！
0       |不是食品安全新闻！
-1       |标题格式有误！


# 使用
``` bash
docker pull bringtree/news_api
docker run -d -p 0.0.0.0:5000:5000 -v /home/bringtree/wordvec/10G_dict.pkl:/home/10G_dict.pkl:ro bringtree/news_api
```

# gpu
``` bash
docker pull bringtree/news_api:gpu
docker run --runtime=nvidia  -d -p 0.0.0.0:5000:5000 -v /home/bringtree/wordvec/10G_dict.pkl:/home/10G_dict.pkl:ro bringtree/news_api:gpu
```
