import requests
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
import pickle
from pprint import pprint
from datetime import datetime

# creating object
obj_now = datetime.now()

# printing the current date and time
print("Start Date/Time: ", obj_now)

# extracting and printing the current
# hour, minute, second and microsecond
print("Current hour =", obj_now.hour)
print("Current minute =", obj_now.minute)
print("Current second =", obj_now.second)

# url: references default USGS API url
# data_url = the actual data url

url = "https://earthquake.usgs.gov/fdsnws/event/1/[query[format=geojson]]"
data_url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2005-1-01&endtime=2006-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url1 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2005-1-01&endtime=2007-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&minmagnitude=3.9"
data_url2 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2006-1-01&endtime=2007-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url3 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2007-1-01&endtime=2008-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url4 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2007-1-01&endtime=2009-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&minmagnitude=3.9"
data_url5 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2008-1-01&endtime=2009-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url6 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2009-1-01&endtime=2010-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url7 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2009-1-01&endtime=2011-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&minmagnitude=3.9"
data_url8 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2010-1-01&endtime=2011-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url9 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2011-1-01&endtime=2012-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url10 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2011-1-01&endtime=2013-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&minmagnitude=3.9"
data_url11 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2012-1-01&endtime=2013-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url12 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2013-1-01&endtime=2014-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url13 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2013-1-01&endtime=2015-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&minmagnitude=3.9"
data_url14 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2014-1-01&endtime=2015-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url15 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2015-1-01&endtime=2016-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url16 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2015-1-01&endtime=2017-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&minmagnitude=3.9"
data_url17 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2016-1-01&endtime=2017-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url18 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2017-1-01&endtime=2018-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url19 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2017-1-01&endtime=2019-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&minmagnitude=3.9"
data_url20 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2018-1-01&endtime=2019-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url21 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2019-1-01&endtime=2020-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
data_url22 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2019-1-01&endtime=2021-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&minmagnitude=3.9"
data_url23 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2020-1-01&endtime=2021-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1&minsig=1&maxmagnitude=3.9"
# json response and data format
response = requests.get(data_url)
response1 = requests.get(data_url1)
response2 = requests.get(data_url2)
response3 = requests.get(data_url3)
response4 = requests.get(data_url4)
response5 = requests.get(data_url5)
response6 = requests.get(data_url6)
response7 = requests.get(data_url7)
response8 = requests.get(data_url8)
response9 = requests.get(data_url9)
response10 = requests.get(data_url10)
response11 = requests.get(data_url11)
response12 = requests.get(data_url12)
response13 = requests.get(data_url13)
response14 = requests.get(data_url14)
response15 = requests.get(data_url15)
response16 = requests.get(data_url16)
response17 = requests.get(data_url17)
response18 = requests.get(data_url18)
response19 = requests.get(data_url19)
response20 = requests.get(data_url20)
response21 = requests.get(data_url21)
response22 = requests.get(data_url22)
response23 = requests.get(data_url23)






data = response.json()
data1 = response1.json()
data2 = response2.json()
data3 = response3.json()
data4 = response4.json()
data5 = response5.json()
data6 = response6.json()
data7 = response7.json()
data8 = response8.json()
data9 = response9.json()
data10 = response10.json()
data11 = response11.json()
data12 = response12.json()
data13 = response13.json()
data14 = response14.json()
data15 = response15.json()
data16 = response16.json()
data17 = response17.json()
data18 = response18.json()
data19 = response19.json()
data20 = response20.json()
data21 = response21.json()
data22 = response22.json()
data23 = response23.json()

data_list = [data, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13,
             data14, data15, data17, data18, data19, data20, data21, data22, data23]
# needed_data = ["magType", "mag", "dmin"]
# mag = data["features"][0]["properties"]["mag"]
# magType = data["features"][0]["properties"]["magType"]
# dmin = data["features"][0]["properties"]["dmin"]

# creates lists for needed data  (RYAN: make an empty tsunami list here)
dmin = []
tsunami = []
mag = []
magType = []
gap = []
rms = []
sig = []
none = []
depth = []
# pprint(data)

#wrapper to extract data from GEOJson
for x in data_list:
    for y in x["features"]:
        if y["properties"]["dmin"] == None:
            none.append((y["properties"]["dmin"]))
        else:
            dmin.append(y["properties"]["dmin"])

        if y["properties"]["mag"] == None:
            none.append((y["properties"]["mag"]))
        else:
            mag.append(y["properties"]["mag"])

        if y["properties"]["magType"] == None:
            none.append((y["properties"]["magType"]))
        else:
            magType.append(y["properties"]["magType"])

        if y["properties"]["gap"] == None:
            none.append((y["properties"]["gap"]))
        else:
            gap.append(y["properties"]["gap"])

        if y["properties"]["rms"] == None:
            none.append((y["properties"]["rms"]))
        else:
            rms.append(y["properties"]["rms"])

        if y["properties"]["sig"] == None:
            none.append((y["properties"]["sig"]))
        else:
            sig.append(y["properties"]["sig"])

        if y["geometry"]["coordinates"][2] == None:
            none.append((y["geometry"]["coordinates"][2]))
        else:
            depth.append((y["geometry"]["coordinates"][2]))


print("Data Intake Done")



# creates a an empty list to lower case all waveforms to correct for inconsistency in data entry
case_sensitive_Waveforms = []
for x in magType:
   case_sensitive_Waveforms.append(x.lower())

# creates dataframe from lists
depth_DF = pd.DataFrame({"Depth": depth})
dmin_Df = pd.DataFrame({"Distance": dmin})
mag_df = pd.DataFrame({"Magnitude": mag})
mag_df["Magnitude"] = mag_df["Magnitude"].abs()
magType_df = pd.DataFrame({"Waveform": case_sensitive_Waveforms})
rms_DF = pd.DataFrame({"Root Mean Square": rms})
gap_DF = pd.DataFrame({"Azimuthal Gap": gap})
sig_DF = pd.DataFrame({"Signature": sig})
tsunami_df = pd.DataFrame({"Tsunami": tsunami})
# print(dmin_Df.max(), dmin_Df.min())
# used to show statistics distribution of prediction variables
# a good distribution of wave forms is important
Waveformtotals = magType_df.value_counts()
Waveformtotals = Waveformtotals.rename("Count")
Waveformtotals.to_csv(path_or_buf="data/Waveformtotals.csv")
# print(magType_df.value_counts())

# merges dataframes made above
Ses_DF = pd.DataFrame(dmin_Df)
Ses_DF = Ses_DF.merge(mag_df, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.merge(magType_df, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.merge(rms_DF, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.merge(gap_DF, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.merge(depth_DF, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.merge(sig_DF, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.dropna()
Ses_DF.to_csv(path_or_buf="Data.csv")
# print(Ses_DF)
print("Dataframe Construction DONE")


# sets variables to be predicted X is trainer Variables and y is target


# (RYAN: Change y = Ses_DF["Waveform"] to y = Ses_DF["Tsuanmi"])

#RYAN: NOTE OF CAUTION: tsunamis are EXTREMELY RARE as a result your data WILL be skewed. If you get a value of over 90% from ML score,
#CONTINUED NOTE: please uncomment my Lines of code below to make sure you have a roughly than 70%-30% total data set size to tsunami generated ratio at minimum

# values = tsunami_df.value_counts()
# values = values[1]
# values = values.values
# values = values.max()
# values = int(values)
# size = tsunami_df.count()
# ratio = (values/size)*100
# print(f"RATIO OF TSUNAMI INCIDENTS:{ratio}")


X = Ses_DF[["Magnitude", "Depth", "Distance","Azimuthal Gap", "Root Mean Square"]]
y = Ses_DF["Waveform"]
pca = PCA(n_components=5)
pca.fit(X)
# test train split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=10)
# model creation
clf = GaussianNB()

print("Model Creation DONE, Beginning Fit")

# fits data
clf.fit(X_train, Y_train)
print("Model Fitting DONE")

# scores model
score = clf.score(X_test, Y_test)
print(f"Train Score:{clf.score(X_train,Y_train)}")
print(f"Accuracy of model: {score*100}%")
print("Model Scoring DONE")





pkl_filename = "GuassNB_pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)
print("Saving DONE, Finished Program at:")

obj_now = datetime.now()

# printing the current date and time
print("Finish Date/Time: ", obj_now)

# extracting and printing the current
# hour, minute, second and microsecond
print("Current hour =", obj_now.hour)
print("Current minute =", obj_now.minute)
print("Current second =", obj_now.second)

# EXTRA CODE FOR FUTURE USE
# score = predicted.score(X,y)
# score = clf.score(X_test,Y_test)
# print(score)
# NB = naive_bayes.GaussianNB()

# plot = plt.plot()