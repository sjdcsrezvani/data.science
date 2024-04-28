# import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline
company = ['GOOGL', 'AMNZ', 'MSFT', 'F8']
revenue = [90, 136, 89, 27]
profit = [40, 2, 34, 12]
plt.bar(company, revenue)  # showing our data as bar chart
plt.bar(company, revenue, label='revenue')
plt.legend()  # it indicates that these bar charts belong to revenue
# let's say we have another data like profit, and we want to add new bars to our chart
plt.bar(company, profit, label='profit')  # it will show the data on the same bar
xpos = np.arange(len(company))  # creating a range of integer for our company list
plt.xticks(xpos, company)  # referring our range to our company list
plt.bar(xpos - 0.2, revenue, label='revenue')
plt.bar(xpos + 0.2, profit, label='profit')  # now we have 2 bars for each company in one chart by just changing value
# of our range
plt.bar(xpos - 0.2, revenue, width=0.4, label='revenue')  # to control width of the bar

plt.barh(xpos - 0.2, revenue, label='revenue')
plt.barh(xpos + 0.2, profit, label='profit')  # it will create horizontal bar chart

# when we are working with histograms we only need one dimension array and y-axis is frequency
blood_sugar = [113, 85, 90, 150, 149, 88, 93, 115, 80, 77, 82, 129]
plt.hist(blood_sugar)  # creating histogram
plt.hist(blood_sugar, bins=3)  # bins= indicates how many bars we have in our histogram
plt.hist(blood_sugar, bins=3, rwidth=0.95)  # now we have a little distance between our bars
plt.hist(blood_sugar, bins=[80,100,125,200], rwidth= 0.95)
plt.hist(blood_sugar, bins=[80,100,125,150], rwidth= 0.95, histtype='step')  # we change histogram style with hist-type=

# let's say we have two samples of data sugar , one for men and one for women
blood_sugar_w= [67,98,89,120,133,150,84,69,79,120,112,100]
plt.hist([blood_sugar,blood_sugar_w])  # now we have histogram with both of the samples
plt.hist([blood_sugar,blood_sugar_w],label=['men','women'])  # have to refer anything in a list for our histogram
plt.hist([blood_sugar,blood_sugar_w],label=['men','women'], orientation='horizontal')  # it will plot it on y-axis


# using pie-chart we want to know each part percentage in the hole thing
exp_vals=[1400,600,300,410,250]
exp_labels=['home rent', 'food', 'phone/internet bill', 'car', 'other utilities']  # we have our expenses here
plt.pie(exp_vals,labels=exp_labels)  # plotting the pie chart
plt.show()  # to remove the detail text above our chart
plt.pie(exp_vals,labels=exp_labels,radius=2)  # changing the radius of our pie chart
plt.axis('equal')  # to create perfect circle pie chart
plt.pie(exp_vals,labels=exp_labels, radius=1, autopct='%0.2f%%')  # showing our percentage in our pie chart and '0.2'
# indicates the decimal point it will show in pie chart
plt.pie(exp_vals,labels=exp_labels, radius=1, autopct='%0.2f%%', shadow=True)  # create a shadow for our chart
plt.pie(exp_vals,labels=exp_labels, radius=1, autopct='%0.2f%%', shadow=True, explode=[0,0.2,0,0,0])  # explode
# attribute : we will choose any part in our pie chart and it will come out like piece of pie by the amount we are
# giving to it
plt.pie(exp_vals,labels=exp_labels, radius=1, autopct='%0.2f%%', shadow=True, explode=[0,0.2,0,0,0],startangle=45)  #
# it will start plotting our pie chart in the given angle
