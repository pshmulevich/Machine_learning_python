Suppose you have a dataset loaded. In my example I am referring to the Boston Housing dataset referred to in my README.md file. You are interested in doing some statistical analysis with it. Here are useful python and pandas commands:

Finding the minimum price of the data

```
minimum_price = np.min(prices)
```

Finding the minimum price of the data in Pandas:

```
minimum_price = prices.min()
```

Finding the maximum price of the data

```
maximum_price = np.max(prices)
```

Finding the maximum price of the data in Pandas:

```
maximum_price = prices.max()
```


Finding the mean price of the data
```
mean_price = np.mean(prices)
```

Finding the mean price of the data in Pandas:

```
mean_price = prices.mean()
```

Finding the median price of the data

```
median_price = np.median(prices)
```

Finding the median price of the data in Pandas:

```
median_price = prices.median()
```

Finding the standard deviation of prices of the data

```
std_price = np.std(prices)
```

Finding the standard deviation of prices of the data in Pandas:

```
std_price = prices.std(ddof=0)
```

Show the calculated statistics

```
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)
```
