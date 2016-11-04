import json
import urllib
from urlparse import urlparse

import httplib2 as http

if __name__=="__main__":
    #parameters
    headers={'AccountKey': 'gTfk1cYNuoVBCnmXoJJ2ag==',
             'UniqueUserID':'c0cdca5a-5c1a-4af1-8e11-24a63cd875a5',
             'accept': 'application/json'}

    #API parameters
    uri='http://datamall2.mytransport.sg/'
    pathnull = 'ltaodataservice/TrafficSpeedBands?$skip='

    #simplify
    BA=CB=FC=IF=HI=GH=HE=EB=AD=DG=EF=DE=0
    sum=BA+CB+FC+IF+HI+GH+HE+EB+AD+DG+EF+DE

    for i in range(0,1167,1):
        if sum==12:
            break

        skip=i*50
        path=pathnull+str(skip)

        target=urlparse(uri+path)
        #print target.geturl()
        method='GET'
        body=''

        #Get handle to http
        h=http.Http()

        #obtain results
        response, content= h.request(
            target.geturl(),
            method,
            body,
            headers
        )

        #Parse JSON to print
        jsonObj=json.loads(content)
        for j in range(0,49,1):

            if jsonObj["value"][j]["LinkID"]=='103020792' and BA==0:
                BA=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandBANT.json","a") as outfile:
                    json.dump(jsonObj["value"][j],outfile,sort_keys=True, indent=4,ensure_ascii=False)
                continue
            if jsonObj["value"][j]["LinkID"] == '103014064' and CB==0:
                CB=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandCBNT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                continue
            if jsonObj["value"][j]["LinkID"] == '103080908' and IF==0:
                IF=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandIFNT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                continue
            if jsonObj["value"][j]["LinkID"] == '103080896' and FC==0:
                FC=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandFCNT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                continue
            if jsonObj["value"][j]["LinkID"] == '103010920' and EF==0:
                EF=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandEFNT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                continue
            if jsonObj["value"][j]["LinkID"] == '103013108' and DE==0:
                DE=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandDENT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                continue
            if jsonObj["value"][j]["LinkID"] == '103055972' and DG==0:
                DG=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandDGNT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                continue
            if jsonObj["value"][j]["LinkID"] == '103086736' and AD==0:
                AD=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandADNT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                continue
            if jsonObj["value"][j]["LinkID"] == '106004344' and GH==0:
                GH=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandGHNT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                continue
            if jsonObj["value"][j]["LinkID"] == '106002986' and HI==0:
                HI=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandHINT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                continue
            if jsonObj["value"][j]["LinkID"] == '103075518' and EB==0:
                EB=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandBENT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandEBNT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                continue
            if jsonObj["value"][j]["LinkID"] == '103075519' and HE==0:
                HE=1
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandEHNT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                with open("/Users/billy/Desktop/FYP/NT/traffic_speedbandHENT.json", "a") as outfile:
                    json.dump(jsonObj["value"][j], outfile, sort_keys=True, indent=4, ensure_ascii=False)
                continue