import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import TimestampType as TimeStamp
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType 

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']

def create_spark_session():
    """
    Create new Spark session

    Parameters: Empty

    Return: a new Spark session object
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark

def process_song_data(spark, input_data, output_data):
    """
    Load data from song_data dataset and extract columns
    for songs and artist tables. Then write the data again to S3.

    Parameters
    ----------
    spark: the spark session
    input_data: the path to the song_data from Udacity.
    output_data: the path to where the S3 bucket will be written.

    Return: None
    """

    # get filepath to song data file
    song_data = input_data + 'song_data/A/*/*/*.json'
    
    # Define data type before read json
    song_schema = StructType([
        StructField("song_id", StringType()),
        StructField("artist_id", StringType()),
        StructField("title", StringType()),
        StructField("artist_latitude", DoubleType()),
        StructField("artist_longitude", DoubleType()),
        StructField("duration", DoubleType()),
        StructField("year", IntegerType()),
    ])
    
    # read song data file
    df = spark.read.json(song_data, schema=song_schema)

    # extract columns to create songs table
    songsTable = df.select(
        ["song_id", "title", "artist_id", "year", "duration"]).dropDuplicates(["song_id"]
    )
    
    # write songs table to S3 files partitioned by year and artist
    songsTable.write.parquet(
        output_data + "songsTable.parquet", partitionBy = ["year", "artist_id"], mode = "overwrite"
    )

    # extract columns to create artists table
    artistsTable = df.select(
        ["artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude"]).dropDuplicates(["artist_id"]
    )
    
    # write artists table to S3 files
    artistsTable.write.parquet(output_data + "artistsTable.parquet", mode = "overwrite")
    
def process_log_data(spark, input_data, output_data):
    """
    Load data from log_data dataset and extract columns
    for users and time tables, songplays json. 
    Then writes the data S3 bucket.

    Parameters
    ----------
    spark: the spark session
    input_data: the path to the log_data on Udacity bucket.
    output_data: My S3 bucket.

    Return: None
    """

    # get filepath to log data file
    log_data = input_data + 'log_data/2018/*/*.json'
    
    # Define data type before read json
    log_schema = StructType([
        StructField("userId", IntegerType()),
        StructField("firstName", StringType()),
        StructField("lastName", StringType()),
        StructField("gender", StringType()),
        StructField("level", StringType()),      
    ])

    # read log data file, filter page == NextSong
    log_df = spark.read.json(log_data, schema=log_schema).filter(log_df.page == "NextSong")
    
    # extract columns for users file    
    usersTable = log_df.select(
        ["userId", "firstName", "lastName", "gender", "level"]).dropDuplicates(["userId"]
    )
    
    # write users table to S3 parquet files
    usersTable.write.parquet(output_data + "usersTable.parquet", mode = "overwrite")

    # Get timestamp column from log_data timestamp column
    get_timestamp = udf(lambda t: datetime.fromtimestamp((t / 1000.0)), TimeStamp())
    log_df = log_df.withColumn("timestamp", getTimestamp(col("ts")))
    
    # extract columns to create time Json
    times_json = log_df.selectExpr(
        "timestamp as start_time",
        "hour(timestamp) as hour",
        "dayofmonth(timestamp) as day",
        "weekofyear(timestamp) as week",
        "month(timestamp) as month",
        "year(timestamp) as year",
        "dayofweek(timestamp) as weekday"
    ).dropDuplicates(["start_time"])
    
    # write time table to S3 files partitioned by year and month
    
    timesTable.write.parquet(
        output_data + "timesTable.parquet", partitionBy = ["year", "month"], mode = "overwrite"
    )

    # read in song data to use for songplays table
    song_data = input_data + "song_data/A/*/*/*.json"
    song_df = spark.read.json(song_data)

    # Extract columns from joined song and log datasets to create songplays table 
    log_df.createOrReplaceTempView("temp_log_data")
    song_df.createOrReplaceTempView("temp_song_data")
    songplays_table = spark.sql("""
        SELECT
            l.timestamp as start_time,
            year(l.timestamp) as year,
            month(l.timestamp) as month,
            l.userId AS user_id,
            l.level as level,
            s.song_id as song_id,
            s.artist_id as artist_id,
            l.sessionId as session_id,
            l.location as location,
            l.userAgent as user_agent
        FROM temp_log_data l
        JOIN 
            temp_song_data s ON l.song = s.title AND l.artist = s.artist_name AND l.length = s.duration

    """)

    # write songplays table to S3 files partitioned by year and month
    
    songplaysTable.write.parquet(
        output_data + "songplaysTable.parquet", partitionBy = ["year", "month"], mode = "overwrite"
    )
    
def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://quyennc2/"  
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)
    
if __name__ == "__main__":
    main()