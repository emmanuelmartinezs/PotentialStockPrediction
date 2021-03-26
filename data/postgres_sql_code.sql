`AWS RDS Connection String`
psql --host=crypto-crystal-ball.cgun02mkb2ko.us-east-1.rds.amazonaws.com --port=5432 --username=postgres --password=******** --dbname=crypto-crystal-ball


`1st Table`
CREATE TABLE "top_coins_price" (
  "ID" SERIAL PRIMARY KEY,
  "Date" timestamp,
  "Open" varchar,
  "High" varchar,
  "Low" varchar,
  "Close" varchar,
  "Volume" varchar,
  "Currency" varchar
);

COPY top_coins_price("ID", "Date", "Open", "High", "Low", "Close", "Volume", "Currency")
FROM 'C:\Users\cburd\class\crypto_crystal_ball\Resources\top_coins_price.csv'
DELIMITER ','
CSV HEADER;

select * from top_coins_price;

`2nd Table`
CREATE TABLE "top_coins_adj_close_price" (
  "ID" int PRIMARY KEY,
  "Adj Close" varchar
);

COPY top_coins_adj_close_price("ID", "Adj Close")
FROM 'C:\Users\cburd\class\crypto_crystal_ball\Resources\top_coins_adj_close_price.csv'
DELIMITER ','
CSV HEADER;

select * from top_coins_adj_close_price;


`Join Tables`
CREATE TABLE top_coins AS
SELECT
  top_coins_price."ID",
  "Date",
  "Open",
  "High",
  "Low",
  "Close",
  "Adj Close",
  "Volume",
  "Currency"
FROM
	top_coins_price
INNER JOIN top_coins_adj_close_price 
    ON top_coins_adj_close_price."ID" = top_coins_price."ID";


`Drop ID for Plotly analysis`
ALTER TABLE top_coins
DROP COLUMN ID;

select*from top_coins

`Export to csv for anlysis portion of project`
COPY top_coins TO 'C:\Users\cburd\class\crypto_crystal_ball\Resources\top_coins.csv' DELIMITER ',' CSV HEADER;
