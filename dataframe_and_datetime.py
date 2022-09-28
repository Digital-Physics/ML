import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# https://docs.python.org/3/library/datetime.html
import sys


def get_user_interaction_counts(search_interaction_df: pd.DataFrame) -> pd.DataFrame:
    """return a new DataFrame of user search counts over the past 1/7/30 days"""
    print("use the .agg() method to apply a function to each dataframe column (or row, axis = 1)")
    print(sys.version)
    # most_recent_date = search_interaction_df.agg({"date": "max"}).collect()[0][0]
    # print(most_recent_date, type(most_recent_date))
    most_recent_date2 = search_interaction_df["date"].agg(func="max")
    print(most_recent_date2, type(most_recent_date2))

    print("turn the string into a datetime object by stripping the time out of the string using strptime()")
    print("indicate the date format. uppercase Y indicates it is a 4 digit year, not a 2 digit year")
    most_recent_date = datetime.strptime(most_recent_date2, "%Y-%m-%d")
    print(most_recent_date, type(most_recent_date))

    last_1_df = get_df_counts_via_datetime_by_user_id(search_interaction_df, most_recent_date, 1)
    last_7_df = get_df_counts_via_datetime_by_user_id(search_interaction_df, most_recent_date, 7)
    last_30_df = get_df_counts_via_datetime_by_user_id(search_interaction_df, most_recent_date, 30)

    print("searches by all users over the last month(30 days) is a superset of the other two user lists, so we'll make it first")
    df_out = pd.DataFrame()
    df_out["month_interaction_count"] = last_30_df
    print(df_out)

    print("We'll then left .merge() the other DataFrames. on='user_id' key. 'how'? a 'left' join")
    df_out = pd.merge(df_out, last_7_df, on="user_id", how="left")
    df_out = pd.merge(df_out, last_1_df, on="user_id", how="left")

    print("rename columns")
    df_out = df_out.rename(columns={"search_term_x": "week_interaction_count", "search_term_y": "day_interaction_count"})

    print("We'll then fill in missing values for those user_ids on the left but not the right of the joins")
    print("can use .fillna() or .replace()")
    # df_out = df_out.fillna(0)
    df_out = df_out.replace(np.nan, 0)
    print(df_out)

    return df_out


def get_df_counts_via_datetime_by_user_id(df_input: pd.DataFrame, date: datetime, days: int) -> pd.DataFrame:
    """get DataFrame counts by filter between two datetimes"""
    start_date = date - timedelta(days=days)
    df_input["date"] = pd.to_datetime(df_input["date"])
    return df_input.where(df_input["date"].between(start_date, date)).groupby("user_id")["search_term"].count()


class Row:
    def __init__(self, date: str, ip_address: str, request_path: str, search_term: str, user_id: str) -> None:
        self.date = date
        self.ip_address = ip_address
        self.request_path = request_path
        self.search_term = search_term
        self.user_id = user_id


data = [Row(date='2021-04-07', ip_address='168.196.83.238', request_path='search', search_term='Saving Private Ryan', user_id='09663ea6'),
        Row(date='2021-04-07', ip_address='116.103.0.64', request_path='search', search_term='Fear and Loathing', user_id='63f84e80'),
        Row(date='2021-04-07', ip_address='9.47.206.231', request_path='search', search_term='Legally Blonde', user_id='31c73683'),
        Row(date='2021-04-07', ip_address='198.45.207.12', request_path='search', search_term='Legally Blonde', user_id='010b4076'),
        Row(date='2021-04-07', ip_address='248.212.242.192', request_path='search', search_term='The Hills Have Eyes', user_id='31c73683'),
        Row(date='2021-04-01', ip_address='65.166.90.163', request_path='search', search_term='Knives Out', user_id='8173164f'),
        Row(date='2021-04-01', ip_address='129.96.138.214', request_path='search', search_term='The Hills Have Eyes', user_id='010b4076'),
        Row(date='2021-04-01', ip_address='226.101.244.94', request_path='search', search_term='Goodwill Hunting', user_id='010b4076'),
        Row(date='2021-04-01', ip_address='5.245.67.32', request_path='search', search_term='Goodwill Hunting', user_id='f77ad2d3'),
        Row(date='2021-04-01', ip_address='167.74.212.249', request_path='search', search_term='Fear and Loathing', user_id='ca7aacf2'),
        Row(date='2021-03-26', ip_address='42.199.5.131', request_path='search', search_term='Fear and Loathing', user_id='010b4076'),
        Row(date='2021-03-26', ip_address='168.226.225.209', request_path='search', search_term='Another Round', user_id='8173164f'),
        Row(date='2021-03-26', ip_address='153.118.84.141', request_path='search', search_term='Saving Private Ryan', user_id='25050522'),
        Row(date='2021-03-26', ip_address='243.62.232.219', request_path='search', search_term='A Few Good Men', user_id='010b4076'),
        Row(date='2021-03-26', ip_address='87.178.83.203', request_path='search', search_term='Fear and Loathing', user_id='ca7aacf2'),
        Row(date='2021-03-13', ip_address='102.173.153.248', request_path='search', search_term='Saving Private Ryan', user_id='cbb81ed7'),
        Row(date='2021-03-13', ip_address='92.127.186.131', request_path='search', search_term='Another Round', user_id='bfb27c75'),
        Row(date='2021-03-13', ip_address='232.131.143.148', request_path='search', search_term='Saving Private Ryan', user_id='63f84e80'),
        Row(date='2021-03-13', ip_address='64.27.52.88', request_path='search', search_term='Booksmart', user_id='bfb27c75'),
        Row(date='2021-03-13', ip_address='131.176.53.166', request_path='search', search_term='Legally Blonde', user_id='31c73683'),
        Row(date='2021-04-02', ip_address='175.22.16.190', request_path='search', search_term='Remember the Titans', user_id='cbb81ed7'),
        Row(date='2021-04-02', ip_address='173.179.94.31', request_path='search', search_term='A Few Good Men', user_id='f77ad2d3'),
        Row(date='2021-04-02', ip_address='208.232.206.73', request_path='search', search_term='Another Round', user_id='8173164f'),
        Row(date='2021-04-02', ip_address='34.61.234.215', request_path='search', search_term='Another Round', user_id='010b4076'),
        Row(date='2021-04-02', ip_address='203.59.254.73', request_path='search', search_term='Legally Blonde', user_id='8173164f'),
        Row(date='2021-03-25', ip_address='224.114.231.124', request_path='search', search_term='Legally Blonde', user_id='31c73683'),
        Row(date='2021-03-25', ip_address='134.155.71.108', request_path='search', search_term='Fear and Loathing', user_id='010b4076'),
        Row(date='2021-03-25', ip_address='95.91.124.141', request_path='search', search_term='The Godfather', user_id='bfb27c75'),
        Row(date='2021-03-25', ip_address='243.254.15.230', request_path='search', search_term='Forrest Gump', user_id='bfb27c75'),
        Row(date='2021-03-25', ip_address='2.175.219.25', request_path='search', search_term='Forrest Gump', user_id='ca7aacf2'),
        Row(date='2021-04-06', ip_address='93.27.12.47', request_path='search', search_term='A Few Good Men', user_id='09663ea6'),
        Row(date='2021-04-06', ip_address='97.252.88.139', request_path='search', search_term='The Shape of Water', user_id='31c73683'),
        Row(date='2021-04-06', ip_address='231.141.233.120', request_path='search', search_term='Knives Out', user_id='cbb81ed7'),
        Row(date='2021-04-06', ip_address='239.96.18.102', request_path='search', search_term='Saving Private Ryan', user_id='25050522'),
        Row(date='2021-04-06', ip_address='167.2.38.217', request_path='search', search_term='Saving Private Ryan', user_id='25050522'),
        Row(date='2021-04-08', ip_address='201.170.241.220', request_path='search', search_term='Scary Movie', user_id='63f84e80'),
        Row(date='2021-04-08', ip_address='172.22.43.102', request_path='search', search_term='Scary Movie', user_id='bfb27c75'),
        Row(date='2021-04-08', ip_address='109.21.46.18', request_path='search', search_term='Legally Blonde', user_id='f77ad2d3'),
        Row(date='2021-04-08', ip_address='96.110.209.211', request_path='search', search_term='The Hills Have Eyes', user_id='31c73683'),
        Row(date='2021-04-08', ip_address='171.221.86.38', request_path='search', search_term='The Avengers', user_id='63f84e80'),
        Row(date='2021-03-15', ip_address='250.151.140.121', request_path='search', search_term='The Avengers', user_id='09663ea6'),
        Row(date='2021-03-15', ip_address='126.196.219.173', request_path='search', search_term='Scary Movie', user_id='63f84e80'),
        Row(date='2021-03-15', ip_address='19.161.152.168', request_path='search', search_term='Fear and Loathing', user_id='09663ea6'),
        Row(date='2021-03-15', ip_address='236.164.237.145', request_path='search', search_term='Legally Blonde', user_id='ca7aacf2'),
        Row(date='2021-03-15', ip_address='197.14.149.131', request_path='search', search_term='Wreck It Ralph', user_id='f77ad2d3'),
        Row(date='2021-03-28', ip_address='244.198.45.48', request_path='search', search_term='A Quiet Place', user_id='cbb81ed7'),
        Row(date='2021-03-28', ip_address='44.236.52.91', request_path='search', search_term='Wreck It Ralph', user_id='09663ea6'),
        Row(date='2021-03-28', ip_address='200.174.238.116', request_path='search', search_term='Fear and Loathing', user_id='63f84e80'),
        Row(date='2021-03-28', ip_address='49.238.225.168', request_path='search', search_term='The Godfather', user_id='cbb81ed7'),
        Row(date='2021-03-28', ip_address='246.113.220.169', request_path='search', search_term='The Godfather', user_id='31c73683'),
        Row(date='2021-03-24', ip_address='61.250.126.82', request_path='search', search_term='Saving Private Ryan', user_id='bfb27c75'),
        Row(date='2021-03-24', ip_address='25.79.139.14', request_path='search', search_term='Scary Movie', user_id='63f84e80'),
        Row(date='2021-03-24', ip_address='42.233.200.183', request_path='search', search_term='Remember the Titans', user_id='09663ea6'),
        Row(date='2021-03-24', ip_address='207.153.224.79', request_path='search', search_term='The Dark Knight', user_id='cbb81ed7'),
        Row(date='2021-03-24', ip_address='161.175.80.18', request_path='search', search_term='The Godfather', user_id='bfb27c75'),
        Row(date='2021-04-03', ip_address='45.23.228.97', request_path='search', search_term='Remember the Titans', user_id='09663ea6'),
        Row(date='2021-04-03', ip_address='217.224.197.5', request_path='search', search_term='Wreck It Ralph', user_id='cbb81ed7'),
        Row(date='2021-04-03', ip_address='196.158.184.95', request_path='search', search_term='The Dark Knight', user_id='bfb27c75'),
        Row(date='2021-04-03', ip_address='142.176.223.171', request_path='search', search_term='Another Round', user_id='09663ea6'),
        Row(date='2021-04-03', ip_address='82.87.47.67', request_path='search', search_term='Wreck It Ralph', user_id='31c73683'),
        Row(date='2021-04-09', ip_address='215.101.13.55', request_path='search', search_term='Knives Out', user_id='8173164f'),
        Row(date='2021-04-09', ip_address='134.96.191.88', request_path='search', search_term='Legally Blonde', user_id='25050522'),
        Row(date='2021-04-09', ip_address='110.91.222.130', request_path='search', search_term='Legally Blonde', user_id='f77ad2d3'),
        Row(date='2021-04-09', ip_address='175.142.165.122', request_path='search', search_term='Another Round', user_id='25050522'),
        Row(date='2021-04-09', ip_address='12.146.59.177', request_path='search', search_term='Saving Private Ryan', user_id='25050522'),
        Row(date='2021-03-19', ip_address='246.103.144.21', request_path='search', search_term='Wreck It Ralph', user_id='bfb27c75'),
        Row(date='2021-03-19', ip_address='141.86.235.227', request_path='search', search_term='The Hills Have Eyes', user_id='63f84e80'),
        Row(date='2021-03-19', ip_address='103.170.254.82', request_path='search', search_term='A Few Good Men', user_id='25050522'),
        Row(date='2021-03-19', ip_address='127.218.83.172', request_path='search', search_term='A Few Good Men', user_id='63f84e80'),
        Row(date='2021-03-19', ip_address='247.145.45.51', request_path='search', search_term='The Godfather', user_id='010b4076'),
        Row(date='2021-03-21', ip_address='17.140.75.208', request_path='search', search_term='The Godfather', user_id='31c73683'),
        Row(date='2021-03-21', ip_address='219.145.138.16', request_path='search', search_term='A Quiet Place', user_id='09663ea6'),
        Row(date='2021-03-21', ip_address='173.106.79.128', request_path='search', search_term='Fear and Loathing', user_id='25050522'),
        Row(date='2021-03-21', ip_address='113.90.88.102', request_path='search', search_term='The Hills Have Eyes', user_id='010b4076'),
        Row(date='2021-03-21', ip_address='57.222.119.243', request_path='search', search_term='Dunkirk', user_id='010b4076'),
        Row(date='2021-04-05', ip_address='238.106.111.182', request_path='search', search_term='The Godfather', user_id='8173164f'),
        Row(date='2021-04-05', ip_address='104.211.161.13', request_path='search', search_term='Wreck It Ralph', user_id='f77ad2d3'),
        Row(date='2021-04-05', ip_address='171.130.103.43', request_path='search', search_term='Remember the Titans', user_id='010b4076'),
        Row(date='2021-04-05', ip_address='207.143.209.62', request_path='search', search_term='Another Round', user_id='31c73683'),
        Row(date='2021-04-05', ip_address='172.252.94.171', request_path='search', search_term='Forrest Gump', user_id='010b4076'),
        Row(date='2021-03-23', ip_address='70.23.38.4', request_path='search', search_term='The Shape of Water', user_id='010b4076'),
        Row(date='2021-03-23', ip_address='224.167.136.127', request_path='search', search_term='Legally Blonde', user_id='63f84e80'),
        Row(date='2021-03-23', ip_address='247.68.194.241', request_path='search', search_term='Legally Blonde', user_id='8173164f'),
        Row(date='2021-03-23', ip_address='185.56.226.96', request_path='search', search_term='Wreck It Ralph', user_id='8173164f'),
        Row(date='2021-03-23', ip_address='242.22.156.139', request_path='search', search_term='Fear and Loathing', user_id='010b4076'),
        Row(date='2021-03-13', ip_address='196.251.119.204', request_path='search', search_term='The Shape of Water', user_id='f77ad2d3'),
        Row(date='2021-03-13', ip_address='166.83.196.125', request_path='search', search_term='Saving Private Ryan', user_id='ca7aacf2'),
        Row(date='2021-03-13', ip_address='92.39.250.88', request_path='search', search_term='The Shape of Water', user_id='09663ea6'),
        Row(date='2021-03-13', ip_address='139.234.73.46', request_path='search', search_term='Legally Blonde', user_id='31c73683'),
        Row(date='2021-03-13', ip_address='137.145.207.20', request_path='search', search_term='Dunkirk', user_id='ca7aacf2'),
        Row(date='2021-03-14', ip_address='135.120.182.203', request_path='search', search_term='Scary Movie', user_id='09663ea6'),
        Row(date='2021-03-14', ip_address='55.159.238.68', request_path='search', search_term='Booksmart', user_id='bfb27c75'),
        Row(date='2021-03-14', ip_address='46.124.19.116', request_path='search', search_term='The Hills Have Eyes', user_id='09663ea6'),
        Row(date='2021-03-14', ip_address='189.80.11.252', request_path='search', search_term='The Godfather', user_id='31c73683'),
        Row(date='2021-03-14', ip_address='146.223.32.149', request_path='search', search_term='The Avengers', user_id='010b4076'),
        Row(date='2021-03-18', ip_address='120.243.8.251', request_path='search', search_term='Saving Private Ryan', user_id='bfb27c75'),
        Row(date='2021-03-18', ip_address='90.150.202.229', request_path='search', search_term='A Few Good Men', user_id='31c73683'),
        Row(date='2021-03-18', ip_address='232.57.227.188', request_path='search', search_term='Fear and Loathing', user_id='25050522'),
        Row(date='2021-03-18', ip_address='98.32.219.23', request_path='search', search_term='Knives Out', user_id='bfb27c75'),
        Row(date='2021-03-18', ip_address='153.45.139.71', request_path='search', search_term='A Quiet Place', user_id='31c73683')]

columns = ["date", "ip_address", "request_path", "search_term", "user_id"]
df = pd.DataFrame(columns=columns)

for i, row_obj in enumerate(data):
    temp_dict = {column: eval(f"row_obj.{column}") for column in columns}
    df.loc[i] = pd.Series(temp_dict)

print(df.head())

get_user_interaction_counts(df)


############Pyspark code below##############
#
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from pyspark.sql import SparkSession
#
# # Building the SparkSession and name
# spark = SparkSession.builder.appName("pandas to spark").getOrCreate()
#
#
# def get_user_interaction_counts(search_interaction_df: pd.DataFrame) -> pd.DataFrame:
#     """return a new DataFrame of user search counts over the past 1/7/30 days.
#     A lot of the dataframe methods/syntax seems  specific to pyspark dataframes, not panda dataframes."""
#     # most_recent_date2 = search_interaction_df["date"].agg(func="max")
#     most_recent_date2 = search_interaction_df.agg({"date": "max"}).collect()[0][0]
#
#     most_recent_date = datetime.strptime(most_recent_date2, "%Y-%m-%d")
#
#     last_1_df = get_df_counts_via_datetime_by_user_id(search_interaction_df, most_recent_date, 1).toPandas()
#     last_7_df = get_df_counts_via_datetime_by_user_id(search_interaction_df, most_recent_date, 7).toPandas()
#     last_30_df = get_df_counts_via_datetime_by_user_id(search_interaction_df, most_recent_date, 30).toPandas()
#     print(last_30_df, type(last_30_df))
#
#     df_out = last_30_df
#
#     df_out = pd.merge(df_out, last_7_df, on="user_id", how="left")
#     df_out = pd.merge(df_out, last_1_df, on="user_id", how="left")
#
#     df_out = df_out.rename(columns={"count_x": "month_interaction_count", "count_y": "week_interaction_count", "count": "day_interaction_count"})
#
#     df_out = df_out.replace(np.nan, 0)
#     df_out = df_out.astype({"month_interaction_count":'int', "week_interaction_count":'int', "day_interaction_count":'int'})
#
#     df_out = spark.createDataFrame(df_out)
#
#     return df_out
#
# def get_df_counts_via_datetime_by_user_id(df_input: pd.DataFrame, date: datetime, days: int) -> pd.DataFrame:
#     """get DataFrame counts by filter between two datetimes"""
#     start_date = date - timedelta(days=days)
#     return df_input.where(df_input["date"].between(start_date, date)).groupBy("user_id").count()
#
