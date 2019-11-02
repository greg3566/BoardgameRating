import requests
from bs4 import BeautifulSoup
import numpy as np
import time
import random

def gatherFromUrl(url,tag, notyet=0):
    i=0
    while True:
        time.sleep(i+1)
        req = requests.get(url)
        if req.status_code==200:
            html = req.text
            soup = BeautifulSoup(html, 'html.parser')
            if(soup.find(name=tag)==None):
                print("error")
            else:
                break
        elif i>=notyet:
            print("Not Yet",req.status_code, i)
        i+=1
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    infos = soup.find_all(name=tag)
    return infos

def gatherUsersFromManyRatedGames(gamepages,userpages):
    # rating이 많이 된 gamepages*100 개의 게임을 평가한 각 최근 userpages*100 명의 유저를 가져옵니다.
    
    userset=set()
    
    for gpi in range(gamepages):
        url="https://boardgamegeek.com/browse/boardgame/page/"+str(gpi+1)+"?sort=numvoters&sortdir=desc"
        print(url)
        infos=gatherFromUrl(url,"tr")
        gamemap = map(lambda x: x.find_all(name="td")[2].find(name="a").get("href").split("/")[2],infos[1:])
        ratnummap = map(lambda x: int(x.find_all(name="td")[5].text),infos[1:])
        lastpage = min(ratnummap)//100 -1
        
        urlhead="https://www.boardgamegeek.com/xmlapi2/thing?id="
        for game in gamemap:
            urlhead+=str(game)+","
        urlhead=urlhead[:-1]
        for upi in range(userpages):
            url=urlhead+"&type=boardgame,boardgameexpansion&ratingcomments=1&page="+str(lastpage-upi)+"&pagesize=100"
            print(url)
            infos=gatherFromUrl(url,"comment")
            usermap=map(lambda x: x.get("username"),infos)
            userset.update(usermap)
    userlist=list(userset)
    random.shuffle(userlist)
    return (userlist[:len(userlist)//10], userlist[len(userlist)//10:])


def gatherRatingsFromUser(userlist,uid,R,I,It,gamelist):
    # userlist에 있는 유저 하나로부터 평가한 게임들과 평가를 가져옵니다.
    
    urlhead="https://www.boardgamegeek.com/xmlapi2/collection?username="
    url=urlhead+userlist[uid]+"&subtype=boardgame&stats=1&rated=1"
    infos=gatherFromUrl(url,"item",notyet=1)
    for item in infos:
        game=item.get("objectid")
        if not (game in gamelist):
            gamelist.append(game)
        gid=gamelist.index(game)
        rating=float(item.find("rating").get("value"))
        R[gid,uid]=rating
        if random.random() <0.001:
            It[gid,uid]=True
        else:
            I[gid,uid]=True
        
def gatherRatingsFromUsers(userlist,iterid,batch,R,I,It,gamelist):
    for uid in range(iterid,iterid+batch):
        gatherRatingsFromUser(userlist,uid,R,I,It,gamelist)
        print(uid,end=" ")
        
def cut(gamelist,userlist,R,I,It,gamedeg=32,userdeg=32):
    games=np.array(gamelist)
    users=np.array(userlist)
    R=R.astype("float32")
    while True:
        ind= np.sum(I+It,axis=1)>=gamedeg
        if np.sum(ind)==len(ind):
            break
        games=games[ind[:len(games)]]
        R=R[ind,:]
        I=I[ind,:]
        It=It[ind,:]
        
        ind= np.sum(I+It,axis=0)>=userdeg
        if np.sum(ind)==len(ind):
            break
        users=users[ind[:len(users)]]
        R=R[:,ind]
        I=I[:,ind]
        It=It[:,ind]
    gamelist=list(games)
    userlist=list(users)
    
    return gamelist,userlist,R,I,It

def gatherInformationOfGames(gamelist):
    namelist=[]
    minplist=[]
    maxplist=[]
    weightlist=[]
    catmeclist=[]
    
    def catmec(x):
        mc=map(lambda y: y.get("id"), x.find_all("link",type="boardgamecategory"))
        mm=map(lambda y: y.get("id"), x.find_all("link",type="boardgamemechanic"))
        l=list(mc)
        l.extend(mm)
        return l
    
    batch=100
    urlhead="https://www.boardgamegeek.com/xmlapi2/thing?id="
    l=len(gamelist)
    for i in range( (l-1)//batch +1 ):
        url=urlhead
        for game in gamelist[batch*i:batch*(i+1)]:
            url+=str(game)+","
        url=url[:-1]
        url+="&type=boardgame,boardgameexpansion&stats=1"
        print(url)
        infos=gatherFromUrl(url,"item")
        namemap=map(lambda x: x.find("name").get("value"),infos)
        minpmap=map(lambda x: float(x.find("minplayers").get("value")),infos)
        maxpmap=map(lambda x: float(x.find("maxplayers").get("value")),infos)
        weightmap=map(lambda x: float(x.find("averageweight").get("value")),infos)
        catmecmap=map(catmec,infos)
        
        namelist.extend(namemap)
        minplist.extend(minpmap)
        maxplist.extend(maxpmap)
        weightlist.extend(weightmap)
        catmeclist.extend(catmecmap)
        
    return namelist, minplist, maxplist, weightlist, catmeclist