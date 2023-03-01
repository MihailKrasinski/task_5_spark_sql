from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, dense_rank, \
    coalesce, unix_timestamp, round, row_number, desc, lower
from pyspark.sql.window import Window
from dotenv import load_dotenv, dotenv_values

spark = SparkSession \
    .builder \
    .master('local[*]') \
    .appName('task_5_spark_sql') \
    .config("spark.jars", "/opt/spark/jars/postgresql-42.2.23.jar") \
    .getOrCreate()


load_dotenv()
creds = dotenv_values(".env")


def get_table_frm_db(table_name: str):
    """
    Function gets table_name and
    returns table from database
    """
    return spark.read \
        .format("jdbc") \
        .option("url", creds['URL']) \
        .option("user", creds['USER']) \
        .option("dbtable", table_name) \
        .option("password", creds['PASSWORD']) \
        .option("driver", "org.postgresql.Driver") \
        .load()


category = get_table_frm_db('category')
film = get_table_frm_db('film')
film_actor = get_table_frm_db('film_actor')
actor = get_table_frm_db('actor')
inventory = get_table_frm_db('inventory')
rental = get_table_frm_db('rental')
film_category = get_table_frm_db('film_category')
payment = get_table_frm_db('payment')
customer = get_table_frm_db('customer')
address = get_table_frm_db('address')
city = get_table_frm_db('city')


# 1. Вывести количество фильмов в каждой категории, отсортировать по убыванию:

# SELECT name AS category_name, COUNT(film_category.film_id) AS films FROM category
# JOIN film_category ON category.category_id = film_category.category_id
# GROUP BY category.name
# ORDER BY films DESC;

result_1 = category.join(film_category, "category_id")\
        .groupBy("name") \
        .agg(count("film_id").alias("films")) \
        .orderBy(desc("films"))

result_1.show()


# 2. Вывести 10 актеров, чьи фильмы большего всего арендовали, отсортировать по убыванию.

# SELECT actor.first_name, actor.last_name, COUNT(rental_date) as rents FROM actor
# JOIN film_actor USING(actor_id)
# JOIN film USING(film_id)
# JOIN inventory USING(film_id)
# JOIN rental USING(inventory_id)
# GROUP BY actor.actor_id
# ORDER BY rents DESC
# LIMIT 10;

result_2 = actor.join(film_actor, "actor_id") \
        .join(film, "film_id") \
        .join(inventory, "film_id") \
        .join(rental, "inventory_id") \
        .groupBy("actor_id", "first_name", "last_name") \
        .agg(count("rental_date").alias("rents")) \
        .orderBy(desc("rents")) \
        .limit(10)

result_2.show()


# 3. Вывести категорию фильмов, на которую потратили больше всего денег.

# SELECT category.name, SUM(payment.amount) as money_spent from category
# JOIN film_category USING(category_id)
# JOIN film USING(film_id)
# JOIN inventory USING(film_id)
# JOIN rental USING(inventory_id)
# JOIN payment USING(rental_id)
# GROUP BY category.name
# ORDER BY money_spent DESC
# LIMIT 1;

result_3 = category.join(film_category, "category_id") \
        .join(film, "film_id") \
        .join(inventory, "film_id") \
        .join(rental, "inventory_id") \
        .join(payment, "rental_id") \
        .groupBy("name") \
        .agg(sum("amount").alias("money_spent")) \
        .orderBy(desc("money_spent")) \
        .limit(1)

result_3.show()


# 4. Вывести названия фильмов, которых нет в inventory. Написать запрос без использования оператора IN.

# SELECT film.title FROM inventory
# RIGHT JOIN film ON inventory.film_id = film.film_id
# WHERE inventory.film_id IS NULL;

result_4 = inventory.join(film, "film_id", "right") \
          .where(inventory.film_id.isNull()) \
          .select("title")

result_4.show()

result_4_way_2 = film \
            .join(inventory, 'film_id', 'left') \
            .filter(col('inventory_id').isNull()) \
            .select('title')
result_4_way_2.show()


# 5. Вывести топ 3 актеров, которые больше всего появлялись в фильмах в категории “Children”.
#    Если у нескольких актеров одинаковое кол-во фильмов, вывести всех.

# SELECT actor.first_name, actor.last_name, COUNT(film.film_id) AS children_films
# FROM film
# JOIN film_category USING(film_id)
# JOIN category USING(category_id)
# JOIN film_actor USING(film_id)
# JOIN actor USING(actor_id)
# WHERE category.name='Children'
# GROUP BY actor.actor_id
# ORDER BY children_films DESC
# FETCH FIRST 3 ROWS WITH TIES;


# result_x = film.join(film_category, "film_id") \
#         .join(category, "category_id") \
#         .join(film_actor, "film_id") \
#         .join(actor, "actor_id") \
#         .where(col("name") == "Children") \
#         .groupBy("actor_id", "first_name", "last_name") \
#         .agg(count("film_id").alias("children_films")) \
#         .orderBy(desc("children_films")) \
#         .limit(3)
#
# result_x.show()

result_5 = actor \
            .join(film_actor, 'actor_id') \
            .join(film, 'film_id') \
            .join(film_category, 'film_id') \
            .join(category, 'category_id') \
            .filter(col('name') == 'Children') \
            .groupBy('actor_id', 'first_name', 'last_name') \
            .agg(count('actor_id').alias('children_films')) \
            .withColumn('rank', dense_rank().over(Window.orderBy(col('children_films').desc()))) \
            .filter(col('rank') < 4) \
            .orderBy(col('children_films').desc())

result_5.show()


# 6. Вывести города с количеством активных и неактивных клиентов (активный — customer.active = 1).
#    Отсортировать по количеству неактивных клиентов по убыванию.

# SELECT city.city, COUNT(customer.active = 1) AS active, COUNT(customer.active = 0) AS non_active FROM city
# JOIN address USING(city_id)
# JOIN customer USING(address_id)
# GROUP BY city.city
# ORDER BY count(customer.active = 0) desc;


result_6 = customer \
            .join(address, 'address_id') \
            .join(city, 'city_id') \
            .groupBy('city') \
            .agg(sum('active').alias('active'), count('active').alias('total_users')) \
            .withColumn('non_active', col('total_users') - col('active')) \
            .select('city', 'non_active', 'active') \
            .orderBy(col('non_active').desc())

result_6.show()


# 7. Вывести категорию фильмов, у которой самое большое кол-во часов суммарной аренды в городах
#    (customer.address_id в этом city),
#    и которые начинаются на букву “a”. То же самое сделать для городов в которых есть символ “-”.
#    Написать все в одном запросе.

# WITH category_rent_count AS
#     (SELECT category.name, SUM(age(rental.return_date,rental.rental_date)) AS rent_time, city.city
#     FROM category
#     JOIN film_category USING(category_id)
#     JOIN film USING(film_id)
#     JOIN inventory USING(film_id)
#     JOIN rental USING(inventory_id)
#     JOIN customer USING(customer_id)
#     JOIN address USING(address_id)
#     JOIN city USING(city_id)
#     GROUP BY category.name, city.city)
# SELECT DISTINCT ON(city) name, rent_time, city
# FROM category_rent_count
# WHERE rent_time IN
#     (SELECT MAX(rent_time) FROM category_rent_count GROUP BY city) AND (city LIKE'%-%' OR LOWER(city) LIKE 'a%')
# ORDER BY city, name;


rental_time = rental \
    .withColumn('sum_hours', (coalesce(unix_timestamp('return_date'),
                                       unix_timestamp('last_update')) - unix_timestamp('rental_date')) / 3600) \
    .select('rental_id', 'inventory_id', 'customer_id', 'sum_hours') \
    .orderBy(col('sum_hours').desc())
#
#
result_7 = category \
        .join(film_category, 'category_id') \
        .join(film, 'film_id') \
        .join(inventory, 'film_id') \
        .join(rental_time, 'inventory_id') \
        .join(customer, 'customer_id') \
        .join(address, 'address_id') \
        .join(city, 'city_id') \
        .filter(col('city').like('A%') | col('city').like(r'%-%')) \
        .groupBy('city_id', 'city', 'category_id', 'name') \
        .agg(round(sum('sum_hours'), 2).alias('sum_hours')) \
        .withColumn('rank', row_number().over(Window.partitionBy(col('city_id')).orderBy(col('sum_hours').desc()))) \
        .filter(col('rank') == 1) \


result_7.show()


