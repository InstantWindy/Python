 
import urllib.request 
import re

def getHtml(url):
   page = urllib.request.urlopen(url)
   html = page.read()
   return html

def getImg(html):
   html = html.decode('utf-8')
   reg = 'src="(.+?\.jpg)" alt='
   imgre = re.compile(reg)
   imglist = re.findall(imgre, html)
   x = 0
   for imgurl in imglist:
       urllib.request.urlretrieve(imgurl, '%s.jpg' % x)
       x+=1
   return imglist

html = getHtml("http://pic.yxdown.com/list/0_0_1.html")

