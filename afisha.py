#!/usr/bin/env python
# coding: utf-8

# # Анализ лояльности пользователей Яндекс Афиши

# ## Этапы выполнения проекта
# 
# ### 1. Загрузка данных и их предобработка
# 
# ---
# 
# **Задача 1.1:** Напишите SQL-запрос, выгружающий в датафрейм pandas необходимые данные. Используйте следующие параметры для подключения к базе данных `data-analyst-afisha`:
# 

# 
# Для выгрузки используйте запрос из предыдущего урока и библиотеку SQLAlchemy.
# 
# Выгрузка из базы данных SQL должна позволить собрать следующие данные:
# 
# - `user_id` — уникальный идентификатор пользователя, совершившего заказ;
# - `device_type_canonical` — тип устройства, с которого был оформлен заказ (`mobile` — мобильные устройства, `desktop` — стационарные);
# - `order_id` — уникальный идентификатор заказа;
# - `order_dt` — дата создания заказа (используйте данные `created_dt_msk`);
# - `order_ts` — дата и время создания заказа (используйте данные `created_ts_msk`);
# - `currency_code` — валюта оплаты;
# - `revenue` — выручка от заказа;
# - `tickets_count` — количество купленных билетов;
# - `days_since_prev` — количество дней от предыдущей покупки пользователя, для пользователей с одной покупкой — значение пропущено;
# - `event_id` — уникальный идентификатор мероприятия;
# - `service_name` — название билетного оператора;
# - `event_type_main` — основной тип мероприятия (театральная постановка, концерт и так далее);
# - `region_name` — название региона, в котором прошло мероприятие;
# - `city_name` — название города, в котором прошло мероприятие.
# 
# ---
# 

# In[1]:


# Используйте ячейки типа Code для вашего кода,
# а ячейки типа Markdown для комментариев и выводов


# In[2]:


# При необходимости добавляйте новые ячейки для кода или текста


# In[3]:


get_ipython().system('pip install sqlalchemy')
#!pip install psycopg2 #### на рабочем компе не устанавливается, хотя в Spyder спокойно работает. На домашнем ноуте - все ок


# In[4]:


get_ipython().system('pip install phik')


# In[5]:


import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from phik import phik_matrix
from scipy import stats

import os
from dotenv import load_dotenv


# In[6]:

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    DB_NAME,

)

# In[8]:


engine = create_engine(connection_string)


# In[9]:


tenge = pd.read_csv('https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv')


# In[10]:


query = '''
SELECT 
	p.user_id, p.device_type_canonical,	p.order_id, p.created_dt_msk::date order_dt, p.created_ts_msk order_ts, p.currency_code, p.revenue, p.tickets_count,
	EXTRACT (DAY FROM p.created_dt_msk - lag(p.created_dt_msk) over(PARTITION BY p.user_id ORDER BY p.created_dt_msk)) days_since_prev,
	p.event_id, e.event_name_code event_name, p.service_name, e.event_type_main, r.region_name, c.city_name
FROM afisha.purchases p JOIN afisha.events e using(event_id) JOIN afisha.city c using(city_id) JOIN afisha.regions r using(region_id)
WHERE device_type_canonical IN ('mobile','desktop') 
	AND event_type_main <> 'фильм'
ORDER BY user_id;
'''


# In[11]:


df = pd.read_sql_query(query, con=engine)


# In[12]:


df.info()
print(df)
tenge.info()
print(tenge)


# ---
# 
# **Задача 1.2:** Изучите общую информацию о выгруженных данных. Оцените корректность выгрузки и объём полученных данных.
# 
# Предположите, какие шаги необходимо сделать на стадии предобработки данных — например, скорректировать типы данных.
# 
# Зафиксируйте основную информацию о данных в кратком промежуточном выводе.
# 
# ---

# Информация о датафреймах приведена выше. 290611 строк в основной выгрузке. 357 - в выгрузке с сайта ЦБ. В целом данные выгружены верно. Типы определены корректно, кроме даты order_dt и data в первом и втором датафрейме. Проверки пропусков осуществлялись на этапе разработки запроса. 

# ---
# 
# ###  2. Предобработка данных
# 
# Выполните все стандартные действия по предобработке данных:
# 
# ---
# 
# **Задача 2.1:** Данные о выручке сервиса представлены в российских рублях и казахстанских тенге. Приведите выручку к единой валюте — российскому рублю.
# 
# Для этого используйте датасет с информацией о курсе казахстанского тенге по отношению к российскому рублю за 2024 год — `final_tickets_tenge_df.csv`. Его можно загрузить по пути `https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv')`
# 
# Значения в рублях представлено для 100 тенге.
# 
# Результаты преобразования сохраните в новый столбец `revenue_rub`.
# 
# ---
# 

# Конвертируем типы данных в полях с датами и объединим два датафрема. После объединения расчитаем значения нового поля - `revenue_rub`.

# In[13]:


df['order_dt'] = pd.to_datetime(df['order_dt'], errors='coerce', format='%Y-%m-%d')
tenge['data'] = pd.to_datetime(tenge['data'], errors='coerce', format='%Y-%m-%d')
# print(df.info())
# print(tenge.info())
# print(df)
# print(tenge)


# In[14]:


main = pd.merge(df, tenge, left_on='order_dt', right_on='data', how='left') #left - чтобы не терять строки
main.loc[main['currency_code'] == 'kzt', 'revenue_rub'] = main['revenue'] / main['nominal'] * main['curs']
main.loc[main['currency_code'] != 'kzt', 'revenue_rub'] = main['revenue']


# In[15]:


main[['currency_code','curs','revenue','revenue_rub']].loc[main['currency_code'] == 'kzt'].head(10)


# In[16]:


main[['currency_code','curs','revenue','revenue_rub']].loc[main['currency_code'] == 'rub'].head(10)


# In[17]:


main.isnull().sum()


# Пропуски - только в поле days-since_prev. Конертация курсов прошла успешно.

# ---
# 
# **Задача 2.2:**
# 
# - Проверьте данные на пропущенные значения. Если выгрузка из SQL была успешной, то пропуски должны быть только в столбце `days_since_prev`.
# - Преобразуйте типы данных в некоторых столбцах, если это необходимо. Обратите внимание на данные с датой и временем, а также на числовые данные, размерность которых можно сократить.
# - Изучите значения в ключевых столбцах. Обработайте ошибки, если обнаружите их.
#     - Проверьте, какие категории указаны в столбцах с номинальными данными. Есть ли среди категорий такие, что обозначают пропуски в данных или отсутствие информации? Проведите нормализацию данных, если это необходимо.
#     - Проверьте распределение численных данных и наличие в них выбросов. Для этого используйте статистические показатели, гистограммы распределения значений или диаграммы размаха.
#         
#         Важные показатели в рамках поставленной задачи — это выручка с заказа (`revenue_rub`) и количество билетов в заказе (`tickets_count`), поэтому в первую очередь проверьте данные в этих столбцах.
#         
#         Если обнаружите выбросы в поле `revenue_rub`, то отфильтруйте значения по 99 перцентилю.
# 
# После предобработки проверьте, были ли отфильтрованы данные. Если были, то оцените, в каком объёме. Сформулируйте промежуточный вывод, зафиксировав основные действия и описания новых столбцов.
# 
# ---

# Пропуски - только в `days_since_prev` - установлено ранее. Типы преобразованы

# In[18]:


main = main.drop(['curs','cdx','data','currency_code','revenue','nominal'], axis=1)
main.head(5)


# In[19]:


print(pd.unique(main['event_type_main']))
print(pd.unique(main['service_name']))
# print(pd.unique(main['tickets_count']))
# print(pd.unique(main['region_name']))
# print(main['city_name'].value_counts().sum())


# Явных существенных ошибок в данных не зафиксировано. В поле event_type_main значение "другое" может использоваться в качестве заглушки. Уникальность городов, регионов, операторов проверялась на этапе разработки запроса. На случай, если в этих полях все жеимеются неявные дубликаты, приведем их в upper case и удалим явные дубликаты

# In[20]:


main['service_name_upper'] = main['service_name'].str.upper()
main['region_name_upper'] = main['region_name'].str.upper()
main['city_name_upper'] = main['city_name'].str.upper()


# In[21]:


d_subset=['user_id','device_type_canonical','order_id','tickets_count','event_id','event_name',
          'service_name_upper','region_name_upper','city_name_upper']
main['is_duplicated'] = main.duplicated(subset=d_subset, keep='first')
print('Дублирующиеся строки:')
print(main['is_duplicated'].value_counts())
main[main['is_duplicated']]


# Дублированных строк нет. Удалим поля с upper_case

# In[22]:


main = main.drop(['service_name_upper','region_name_upper','city_name_upper','is_duplicated'], axis=1)
main.head(5)


# Проверьте распределение численных данных и наличие в них выбросов. Для этого используйте статистические показатели, гистограммы распределения значений или диаграммы размаха.
# 
# Важные показатели в рамках поставленной задачи — это выручка с заказа (revenue_rub) и количество билетов в заказе (tickets_count), поэтому в первую очередь проверьте данные в этих столбцах.
# 
# Если обнаружите выбросы в поле revenue_rub, то отфильтруйте значения по 99 перцентилю.
# 
# После предобработки проверьте, были ли отфильтрованы данные. Если были, то оцените, в каком объёме. Сформулируйте промежуточный вывод, зафиксировав основные действия и описания новых столбцов.

# Сводные таблицы cо средним, минимальным, максимальным значениями (для экономии места и читаемости вывод некоторых представлений сводных таблиц закомментирован):

# In[23]:


main['ticket_price'] = main['revenue_rub']/main['tickets_count']
main_pivot_event = main.pivot_table(values='revenue_rub', index='event_type_main', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_region = main.pivot_table(values='revenue_rub', index='region_name', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_city = main.pivot_table(values='revenue_rub', index='city_name', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_operator = main.pivot_table(values='revenue_rub', index='service_name', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_event[['min','max','mean','count','sum','median']]
# print(main_pivot_region[['min','max','mean']])
# print(main_pivot_city[['min','max','mean']])
# print(main_pivot_operator[['min','max','mean']])


# In[24]:


main_pivot_event_t = main.pivot_table(values='tickets_count', index='event_type_main', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_region_t = main.pivot_table(values='tickets_count', index='region_name', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_city_t = main.pivot_table(values='tickets_count', index='city_name', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_operator_t = main.pivot_table(values='tickets_count', index='service_name', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_event_t[['min','max','mean','count','sum','median']]
# print(main_pivot_region[['min','max','mean']])
# print(main_pivot_city[['min','max','mean']])
# print(main_pivot_operator[['min','max','mean']])


# In[25]:


main_pivot_event_tp = main.pivot_table(values='ticket_price', index='event_type_main', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_region_tp = main.pivot_table(values='ticket_price', index='region_name', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_city_tp = main.pivot_table(values='ticket_price', index='city_name', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_operator_tp = main.pivot_table(values='ticket_price', index='service_name', aggfunc=['mean','min','max','count','sum','median'])
main_pivot_event_tp[['min','max','mean','count','sum','median']]


# Присутствуют выбросы - очень дорогие заказы по 81 тыс. руб. (уже позже посчитал цену билета - 21.8 тыс. руб., также можно ситать выбросом - т.е. это не влияние большого количества билетов в заказе.)

# In[26]:


a = main_pivot_event[['median','mean']]
b = main_pivot_event[['max']]
a.sort_values(by=a.columns[0],ascending=False).plot(kind='bar', xlabel='Тип события', ylabel='Стоимость, руб.', legend=True
                 , rot=90, 
    title='Медианная и средняя величина заказа', 
    figsize=(14,5),
    fontsize = 12,
    color = ['royalblue','navy'],
    edgecolor='black'
    )
plt.show()
b.sort_values(by=b.columns[0],ascending=False).plot(kind='bar', xlabel='Тип события', ylabel='Стоимость, руб.', legend=False
                 , rot=90, 
    title='Максимальная величина заказа', 
    figsize=(14,5),
    fontsize = 12,
    color = ['royalblue','navy'],
    edgecolor='black'
    )
plt.show()


# In[27]:


#Гистограммы
plt.figure(figsize=(14,10))
sns.histplot(data=main, x='revenue_rub', bins=80, kde=True, stat='frequency', color='navy')
plt.xlabel('revenue_rub')
plt.ylabel('frequency')
plt.show()


# In[28]:


plt.figure(figsize=(14, 10)) 
# sns.boxplot(data=main, y='event_type_main', x='revenue_rub', showfliers=True)
sns.boxplot(data=main, y='event_type_main', x='revenue_rub', showfliers=False)
plt.rc('font',size=14)
plt.grid(linestyle='--', color='gray', alpha=0.5)
plt.ylabel('Тип события')
plt.xlabel('Выручка')
plt.show()
sns.boxplot(data=main, y='event_type_main', x='revenue_rub', showfliers=True)
plt.show()


# Отфильтруем датафрейм по 99 квантилю по полю revenue_rub

# In[29]:


threshold = main['revenue_rub'].quantile(0.99)
# Фильтруем: оставляем только строки, где value ≤ 99-му перцентилю
main_f = main[main['revenue_rub'] <= threshold]


# In[30]:


main['revenue_rub'].plot(kind='hist', bins=222)
plt.axvline(threshold, color='red', linestyle='--', label='99-й перцентиль')
plt.legend()


# In[31]:


print('Отфильтровано строк: ',main.shape[0]-main_f.shape[0])


# Посмотрим, как поменялась гистограмма (график по городам и регионам аналогичен, но из-за большого количества сущностей плохо читаем, плюс экономлю время):

# In[32]:


main_f_pivot_event = main_f.pivot_table(values='revenue_rub', index='event_type_main', aggfunc=['mean','min','max','count','sum','median'])
c = main_f_pivot_event[['median','mean']]
d = main_f_pivot_event[['max']]
c.sort_values(by=c.columns[0],ascending=False).plot(kind='bar', xlabel='Тип события', ylabel='Стоимость, руб.', legend=True
                 , rot=90, 
    title='Медианная и средняя величина заказа', 
    figsize=(14,5),
    fontsize = 12,
    color = ['royalblue','navy'],
    edgecolor='black'
    )
plt.show()
d.sort_values(by=d.columns[0],ascending=False).plot(kind='bar', xlabel='Тип события', ylabel='Стоимость, руб.', legend=False
                 , rot=90, 
    title='Максимальная величина заказа', 
    figsize=(14,5),
    fontsize = 12,
    color = ['royalblue','navy'],
    edgecolor='black'
    )
plt.show()


# После фильтрации по 99 квантилю распределение данных по заказам стало более равномерным по разным типам событий.

# ---
# 
# ### 3. Создание профиля пользователя
# 
# В будущем отдел маркетинга планирует создать модель для прогнозирования возврата пользователей. Поэтому сейчас они просят вас построить агрегированные признаки, описывающие поведение и профиль каждого пользователя.
# 
# ---
# 
# **Задача 3.1.** Постройте профиль пользователя — для каждого пользователя найдите:
# 
# - дату первого и последнего заказа;
# - устройство, с которого был сделан первый заказ;
# - регион, в котором был сделан первый заказ;
# - билетного партнёра, к которому обращались при первом заказе;
# - жанр первого посещённого мероприятия (используйте поле `event_type_main`);
# - общее количество заказов;
# - средняя выручка с одного заказа в рублях;
# - среднее количество билетов в заказе;
# - среднее время между заказами.
# 
# После этого добавьте два бинарных признака:
# 
# - `is_two` — совершил ли пользователь 2 и более заказа;
# - `is_five` — совершил ли пользователь 5 и более заказов.
# 
# **Рекомендация:** перед тем как строить профиль, отсортируйте данные по времени совершения заказа.
# 
# ---
# 

# In[33]:


#backup датафрейма
df = main
# Сначала сортируем по user_id и времени заказа (order_ts)
df_sorted = df.sort_values(['user_id', 'order_ts']).reset_index(drop=True)

# Группируем по user_id и агрегируем
profile = df_sorted.groupby('user_id', sort=False).agg(
    first_order_dt=('order_dt', 'min'),          # дата первого заказа
    last_order_dt=('order_dt', 'max'),           # дата последнего заказа
    
    # Первый заказ: берём значения первой строки в группе (после сортировки)
    first_device=('device_type_canonical', 'first'),
    first_region=('region_name', 'first'),
    first_service=('service_name', 'first'),
    first_event_type=('event_type_main', 'first'),
    
    total_orders=('order_id', 'count'),          # общее количество заказов
    avg_revenue_rub=('revenue_rub', 'mean'),     # средняя выручка в рублях
    avg_tickets_per_order=('tickets_count', 'mean'),  # среднее количество билетов
    avg_days_between_orders=('days_since_prev', 'mean')  # среднее время между заказами (NULL/NaN игнорируются
).reset_index()

# Добавляем бинарные признаки
profile['is_two'] = (profile['total_orders'] >= 2).astype(int)
profile['is_five'] = (profile['total_orders'] >= 5).astype(int)

# (опционально) приведём типы, если нужно
# profile['first_order_dt'] = pd.to_datetime(profile['first_order_dt'])
# profile['last_order_dt'] = pd.to_datetime(profile['last_order_dt'])


# In[34]:


profile.info()
profile


# ---
# 
# **Задача 3.2.** Прежде чем проводить исследовательский анализ данных и делать выводы, важно понять, с какими данными вы работаете: насколько они репрезентативны и нет ли в них аномалий.
# 
# Используя данные о профилях пользователей, рассчитайте:
# 
# - общее число пользователей в выборке;
# - среднюю выручку с одного заказа;
# - долю пользователей, совершивших 2 и более заказа;
# - долю пользователей, совершивших 5 и более заказов.
# 
# Также изучите статистические показатели:
# 
# - по общему числу заказов;
# - по среднему числу билетов в заказе;
# - по среднему количеству дней между покупками.
# 
# По результатам оцените данные: достаточно ли их по объёму, есть ли аномальные значения в данных о количестве заказов и среднем количестве билетов?
# 
# Если вы найдёте аномальные значения, опишите их и примите обоснованное решение о том, как с ними поступить:
# 
# - Оставить и учитывать их при анализе?
# - Отфильтровать данные по какому-то значению, например, по 95-му или 99-му перцентилю?
# 
# Если вы проведёте фильтрацию, то вычислите объём отфильтрованных данных и выведите статистические показатели по обновлённому датасету.

# In[35]:


user_count = profile['user_id'].value_counts().sum()
avg_order_rev = (profile['total_orders']*profile['avg_revenue_rub']).sum()/(profile['total_orders'].sum())
is_two_share = profile['is_two'].sum()/profile.shape[0]
is_five_share = profile['is_five'].sum()/profile.shape[0]
print('Общее число пользователей в выборке: ', user_count)
print('Средняя выручка с одного заказа: ', round(avg_order_rev,2), 'руб.')
print('Доля пользователей, совершивших 2 и более заказа: ', round(is_two_share,2))
print('Доля пользователей, совершивших 5 и более заказов: ', round(is_five_share,2))


# In[36]:


# Также изучите статистические показатели:
# по общему числу заказов; total_orders
# по среднему числу билетов в заказе; avg_tickets_per_order
# по среднему количеству дней между покупками. avg_days_between_orders


# In[37]:


print('total_orders:')
print('Мин:',     profile['total_orders'].min())
print('Макс:',    profile['total_orders'].max())
print('Среднее:', profile['total_orders'].mean())
print('Медиана:', profile['total_orders'].median())
print()
print('avg_tickets_per_order:')
print('Мин:',     profile['avg_tickets_per_order'].min())
print('Макс:',    profile['avg_tickets_per_order'].max())
print('Среднее:', profile['avg_tickets_per_order'].mean())
print('Медиана:', profile['avg_tickets_per_order'].median())
print()
print('avg_days_between_orders:')
print('Мин:',     profile['avg_days_between_orders'].min())
print('Макс:',    profile['avg_days_between_orders'].max())
print('Среднее:', profile['avg_days_between_orders'].mean())
print('Медиана:', profile['avg_days_between_orders'].median())


# In[38]:


profile_sorted = profile.sort_values(['total_orders'], ascending = False)#.reset_index(drop=True)
profile_sorted


# In[39]:


plt.figure(figsize=(7, 2)) 
plt.rc('font',size=14)
plt.grid(linestyle='--', color='gray', alpha=0.5)
sns.boxplot(data=profile, x='total_orders', showfliers=False)
plt.xlabel('Количество заказов')
plt.show()
plt.figure(figsize=(7, 2)) 
plt.rc('font',size=14)
plt.grid(linestyle='--', color='gray', alpha=0.5)
sns.boxplot(data=profile, x='total_orders', showfliers=True)
plt.xlabel('Количество заказов')
plt.show()

plt.figure(figsize=(7, 2)) 
plt.rc('font',size=14)
plt.grid(linestyle='--', color='gray', alpha=0.5)
sns.boxplot(data=profile, x='avg_tickets_per_order', showfliers=True)
plt.xlabel('Среднее число билетов в заказе')
plt.show()

plt.figure(figsize=(7, 2)) 
plt.rc('font',size=14)
plt.grid(linestyle='--', color='gray', alpha=0.5)
sns.boxplot(data=profile, x='avg_days_between_orders', showfliers=True)
plt.xlabel('Среднее количество дней между покупками')
plt.show()


# Боксплоты со включенной опцией отображения выбросов (showfliers=True) показывают наличие выбросов по всем исследумеым полям. Однако, представляется, что выбросы в полях "среднее число билетов в заказе" и "среднее количество дней" можно оставить, т.к. они, вероятно, не являются аномалией (либо являются, но не во всех случаях): кто-то может покупать 5-12 билетов на компанию из нескольких человек, посещать мероприятия реже 50 дней.  
# Обращает на себя внимание наличие пользователей с большим количеством заказов. Обычное количество заказов - до 5. Иногда - 11. Тем не менее, существуют пользователи с заказами в несколько тысяч (до 10) билетов. Вероятно, это корпоративные покупатели, агентства и т.п.  Скорее всего, они не нужны в исследовании и их можно исключить. Последовательно отфильтруем датафрейм по 99 и 95 процентилю. После фильтрации посмотрим, как изменятся боксплоты.

# In[40]:


threshold99 = profile['total_orders'].quantile(0.99)
threshold95 = profile['total_orders'].quantile(0.95)


# In[41]:


profile99 = profile[profile['total_orders'] <= threshold99]
profile95 = profile[profile['total_orders'] <= threshold95]


# In[42]:


print(f'Удалено строк, 99% {profile.shape[0]-profile99.shape[0]}. Осталось {profile99.shape[0]} строк.')
print(f'Удалено строк, 95% {profile.shape[0]-profile95.shape[0]}. Осталось {profile95.shape[0]} строк.')


# In[43]:


plt.figure(figsize=(7, 2)) 
plt.rc('font',size=14)
plt.grid(linestyle='--', color='gray', alpha=0.5)
sns.boxplot(data=profile, x='total_orders', showfliers=False)
plt.xlabel('Количество заказов')
plt.show()
plt.figure(figsize=(7, 2)) 
plt.rc('font',size=14)
plt.grid(linestyle='--', color='gray', alpha=0.5)
sns.boxplot(data=profile, x='total_orders', showfliers=True)
plt.xlabel('Количество заказов')
plt.show()

plt.figure(figsize=(7, 2)) 
plt.rc('font',size=14)
plt.grid(linestyle='--', color='gray', alpha=0.5)
sns.boxplot(data=profile99, x='total_orders', showfliers=True)
plt.xlabel('Количество заказов. Фильтр по 99% перцетилю')
plt.show()

plt.figure(figsize=(7, 2)) 
plt.rc('font',size=14)
plt.grid(linestyle='--', color='gray', alpha=0.5)
sns.boxplot(data=profile95, x='total_orders', showfliers=True)
plt.xlabel('Количество заказов. Фильтр по 95% перцетилю')
plt.show()


# In[44]:


#гистограмма95
plt.figure(figsize=(8,5))
sns.histplot(data=profile95, x='total_orders', stat='count', bins=40, kde=True)
plt.fontsize=14
plt.title('Распределение количества заказов после фильтрации по 95 перцетилю')
plt.xlabel('Количество заказов')
plt.ylabel('Количество пользователей')
plt.show()


# После фильтрации выбросы по количеству заказов не пропадают полностью, но их количество существенно сокращается, и, как видно из гистограммы выше, в целом таких пользователей немного. Можно оставаить фильтрацию по 95% перцентилю. Доступных данных достаточно для анализа.

# In[45]:


df = profile95 #сохраняем финальный датафрейм для анализа
df.head(5)


# In[46]:


print('Обновленные данные после фильтрации по 95 процентилю:')
print()
user_count = profile95['user_id'].value_counts().sum()
avg_order_rev = (profile95['total_orders']*profile95['avg_revenue_rub']).sum()/(profile95['total_orders'].sum())
is_two_share = profile95['is_two'].sum()/profile95.shape[0]
is_five_share = profile95['is_five'].sum()/profile95.shape[0]
print('Общее число пользователей в выборке: ', user_count)
print('Средняя выручка с одного заказа: ', round(avg_order_rev,2), 'руб.')
print('Доля пользователей, совершивших 2 и более заказа: ', round(is_two_share,2))
print('Доля пользователей, совершивших 5 и более заказов: ', round(is_five_share,2))
print('total_orders:')
print('Мин:',     profile95['total_orders'].min())
print('Макс:',    profile95['total_orders'].max())
print('Среднее:', profile95['total_orders'].mean())
print('Медиана:', profile95['total_orders'].median())
print()
print('avg_tickets_per_order:')
print('Мин:',     profile95['avg_tickets_per_order'].min())
print('Макс:',    profile95['avg_tickets_per_order'].max())
print('Среднее:', profile95['avg_tickets_per_order'].mean())
print('Медиана:', profile95['avg_tickets_per_order'].median())
print()
print('avg_days_between_orders:')
print('Мин:',     profile95['avg_days_between_orders'].min())
print('Макс:',    profile95['avg_days_between_orders'].max())
print('Среднее:', profile95['avg_days_between_orders'].mean())
print('Медиана:', profile95['avg_days_between_orders'].median())


# ---
# 
# ### 4. Исследовательский анализ данных
# 
# Следующий этап — исследование признаков, влияющих на возврат пользователей, то есть на совершение повторного заказа. Для этого используйте профили пользователей.

# 
# 
# #### 4.1. Исследование признаков первого заказа и их связи с возвращением на платформу
# 
# Исследуйте признаки, описывающие первый заказ пользователя, и выясните, влияют ли они на вероятность возвращения пользователя.
# 
# ---
# 
# **Задача 4.1.1.** Изучите распределение пользователей по признакам.
# 
# - Сгруппируйте пользователей:
#     - по типу их первого мероприятия;
#     - по типу устройства, с которого совершена первая покупка;
#     - по региону проведения мероприятия из первого заказа;
#     - по билетному оператору, продавшему билеты на первый заказ.
# - Подсчитайте общее количество пользователей в каждом сегменте и их долю в разрезе каждого признака. Сегмент — это группа пользователей, объединённых определённым признаком, то есть объединённые принадлежностью к категории. Например, все клиенты, сделавшие первый заказ с мобильного телефона, — это сегмент.
# - Ответьте на вопрос: равномерно ли распределены пользователи по сегментам или есть выраженные «точки входа» — сегменты с наибольшим числом пользователей?
# 
# ---
# 

# Рассчитаем сводные таблицы по группам пользователей. Учитывая задания из следующей задачи сразу посчитаем долю пользователей, совершивших два и более заказов

# In[47]:


pt_fe = pd.pivot_table(data=df, index='first_event_type', values=['total_orders'], columns='is_two', aggfunc=['count'] )
df_fe = pd.DataFrame(pt_fe)
df_fe[('count', 'total_orders', 'total')] = df_fe[('count', 'total_orders', 0)]+df_fe[('count', 'total_orders', 1)]
df_fe[('count', 'total_orders', 'share2')] = df_fe[('count', 'total_orders', 1)]/(df_fe[('count', 'total_orders', 0)]+df_fe[('count', 'total_orders', 1)])
df_fe[('count', 'total_orders', 'share_tot')] = (df_fe[('count', 'total_orders', 0)]+df_fe[('count', 'total_orders', 1)])/(df_fe[('count', 'total_orders', 'total')].sum())


# In[48]:


pt_fd = pd.pivot_table(data=df, index='first_device', values=['total_orders'], columns='is_two', aggfunc=['count'] )
df_fd = pd.DataFrame(pt_fd)
df_fd[('count', 'total_orders', 'total')] = df_fd[('count', 'total_orders', 0)]+df_fd[('count', 'total_orders', 1)]
df_fd[('count', 'total_orders', 'share2')] = df_fd[('count', 'total_orders', 1)]/(df_fd[('count', 'total_orders', 0)]+df_fd[('count', 'total_orders', 1)])
df_fd[('count', 'total_orders', 'share_tot')] = (df_fd[('count', 'total_orders', 0)]+df_fd[('count', 'total_orders', 1)])/(df_fd[('count', 'total_orders', 'total')].sum())


# In[49]:


pt_fr = pd.pivot_table(data=df, index='first_region', values=['total_orders'], columns='is_two', aggfunc=['count'] )
df_fr = pd.DataFrame(pt_fr)
df_fr[('count', 'total_orders', 'total')] = df_fr[('count', 'total_orders', 0)]+df_fr[('count', 'total_orders', 1)]
df_fr[('count', 'total_orders', 'share2')] = df_fr[('count', 'total_orders', 1)]/(df_fr[('count', 'total_orders', 0)]+df_fr[('count', 'total_orders', 1)])
df_fr[('count', 'total_orders', 'share_tot')] = (df_fr[('count', 'total_orders', 0)]+df_fr[('count', 'total_orders', 1)])/(df_fr[('count', 'total_orders', 'total')].sum())
df_fr=df_fr.sort_values(by=df_fr.columns[2], ascending=False)#.head(10)
df_fr10=df_fr.sort_values(by=df_fr.columns[2], ascending=False).head(10)


# In[50]:


pt_fs = pd.pivot_table(data=df, index='first_service', values=['total_orders'], columns='is_two', aggfunc=['count'] )
df_fs = pd.DataFrame(pt_fs)
df_fs[('count', 'total_orders', 'total')] = df_fs[('count', 'total_orders', 0)]+df_fs[('count', 'total_orders', 1)]
df_fs[('count', 'total_orders', 'share2')] = df_fs[('count', 'total_orders', 1)]/(df_fs[('count', 'total_orders', 0)]+df_fs[('count', 'total_orders', 1)])
df_fs[('count', 'total_orders', 'share_tot')] = (df_fs[('count', 'total_orders', 0)]+df_fs[('count', 'total_orders', 1)])/(df_fs[('count', 'total_orders', 'total')].sum())
df_fs=df_fs.sort_values(by=df_fs.columns[2], ascending=False)#.head(10)
df_fs10=df_fs.sort_values(by=df_fs.columns[2], ascending=False).head(10)


# In[51]:


df_fe #первые мероприятия


# In[52]:


df_fd #первое устройство


# In[53]:


df_fr #первый регион


# In[54]:


df_fs.head(10)#первый оператор


# *Подсчитайте общее количество пользователей в каждом сегменте и их долю в разрезе каждого признака. Сегмент — это группа пользователей, объединённых определённым признаком, то есть объединённые принадлежностью к категории. Например, все клиенты, сделавшие первый заказ с мобильного телефона, — это сегмент.*
# 
# Количество пользователей каждого сегмента - поле `total` в сводных таблицах выше. Доля в разрезе каждого признака - поле `share_tot`
# 
# *Ответьте на вопрос: равномерно ли распределены пользователи по сегментам или есть выраженные «точки входа» — сегменты с наибольшим числом пользователей?*
# 
# Пользователи распределены не линейно. По каждому сегменту есть выраженные "точки входа". Так. например, по `признаку устройства` - это `mobile`, по `типу мероприятия` - `концерты` и `"другое"`, наиболее активный `оператор` - `"Билеты без проблем"`. По `региону` - `Каменевский регион`.  

# ---
# 
# **Задача 4.1.2.** Проанализируйте возвраты пользователей:
# 
# - Для каждого сегмента вычислите долю пользователей, совершивших два и более заказа.
# - Визуализируйте результат подходящим графиком. Если сегментов слишком много, то поместите на график только 10 сегментов с наибольшим количеством пользователей. Такое возможно с сегментами по региону и по билетному оператору.
# - Ответьте на вопросы:
#     - Какие сегменты пользователей чаще возвращаются на Яндекс Афишу?
#     - Наблюдаются ли успешные «точки входа» — такие сегменты, в которых пользователи чаще совершают повторный заказ, чем в среднем по выборке?
# 
# При интерпретации результатов учитывайте размер сегментов: если в сегменте мало пользователей (например, десятки), то доли могут быть нестабильными и недостоверными, то есть показывать широкую вариацию значений.
# 
# ---
# 

# In[55]:


df_fe[('count', 'total_orders', 'share2')].sort_values(ascending=False).plot(
    kind='bar', 
    xlabel='Тип события', 
    ylabel='Доля', 
    legend=False,
    rot=45, 
    title='Доля пользователей, совершивших два и более заказа', 
    figsize=(10,4),
    fontsize = 12,
    color = ['royalblue'],
    edgecolor='black'
    )
plt.show()

df_fe[('count', 'total_orders', 'share_tot')].sort_values(ascending=False).plot(
    kind='bar', 
    xlabel='Тип события', 
    ylabel='Доля', 
    legend=False,
    rot=45, 
    title='Доля пользователей по сегментам', 
    figsize=(10,4),
    fontsize = 12,
    color = ['navy'],
    edgecolor='black'
    )
plt.show()

print('Медиана: ',df_fe[('count', 'total_orders', 'share2')].median())
print('Среднее: ',df_fe[('count', 'total_orders', 'share2')].mean())
print(df_fe.sort_values(by=df_fe.columns[3], ascending=False))


# По признаку первое посещенное меропритие чаще всего возвращаются пользователи, посетившие `Выставки`, `Театр`, `Концерты`. При этом доля `Выставок` в общем количестве довольно низкая, что может указывать на нестабильность результата по этому параметру.

# In[56]:


##попробуем другой формат представления графиков
df_fd[[('count', 'total_orders', 'share2'),('count', 'total_orders', 'share_tot')]].sort_values(by=df_fd.columns[3], ascending=False).plot(
    kind='bar', 
    xlabel='Тип события', 
    ylabel='Доля', 
    legend=True,
    rot=45, 
    title='Доля пользователей, совершивших два и более заказа', 
    figsize=(10,4),
    fontsize = 12,
    color = ['royalblue','navy'],
    edgecolor='black'
    )
plt.legend(labels=['Доля 2 заказа и более', 'Доля в сегменте'])  # новые подписи
plt.show()

print('Медиана: ',df_fd[('count', 'total_orders', 'share2')].median())
print('Среднее: ',df_fd[('count', 'total_orders', 'share2')].mean())
print(df_fd.sort_values(by=df_fd.columns[3], ascending=False))


# Пользователи, сделавшие первый заказ через `desktop` возвращаются лишь немного чаще, но их почти в 5 раз меньше.

# In[57]:


df_fr10[[('count', 'total_orders', 'share2'),('count', 'total_orders', 'share_tot')]].head(10).sort_values(by=df_fr.columns[3], ascending=False).plot(
    kind='bar', 
    xlabel='Тип события', 
    ylabel='Доля', 
    legend=True,
    rot=45, 
    title='Доля пользователей, совершивших два и более заказа', 
    figsize=(10,4),
    fontsize = 12,
    color = ['royalblue','navy'],
    edgecolor='black'
    )
plt.legend(labels=['Доля 2 заказа и более', 'Доля в сегменте'])  # новые подписи
plt.show()

print('Медиана: ',df_fr[('count', 'total_orders', 'share2')].median())
print('Среднее: ',df_fr[('count', 'total_orders', 'share2')].mean())
print(df_fr10.sort_values(by=df_fr10.columns[3], ascending=False))


# Наиболее активные регионы представлены в таблице выше в порядке снижения доли пользователей, совершивших не менее двух заказов. Наибольшая доля вернувшихся пользователей - в Шанурском регионе, при этом в Каменевском регионе - наибольшее количество пользователей, но доля пользователей с 2 и более заказами - 6я по счету. 

# In[58]:


df_fs10[[('count', 'total_orders', 'share2'),('count', 'total_orders', 'share_tot')]].head(10).sort_values(by=df_fs.columns[3], ascending=False).plot(
    kind='bar', 
    xlabel='Тип события', 
    ylabel='Доля', 
    legend=True,
    rot=45, 
    title='Доля пользователей, совершивших два и более заказа', 
    figsize=(10,4),
    fontsize = 12,
    color = ['royalblue','navy'],
    edgecolor='black'
    )
plt.legend(labels=['Доля 2 заказа и более', 'Доля в сегменте'])  # новые подписи
plt.show()

print('Медиана: ',df_fs[('count', 'total_orders', 'share2')].median())
print('Среднее: ',df_fs[('count', 'total_orders', 'share2')].mean())
print(df_fs10.sort_values(by=df_fs10.columns[3], ascending=False))


# У всех операторов из топ-10 по количеству пользователей доля повторных заказов выше среднего. Самые лояльные пользователи - у оператора `Край билетов`, но доля рынка последнего на фоне остальных - не велика. Результаты по таким провайдерам могут быть не стаблильными, варьироваться в зависимости от каких-либо факторов. Например, от сезона. 
# 
# Поэтому, из тех операторов, у которых доля пользователей выше 10%, самая большая доля клиентов с 2 и более заказами - у оператора `Весь в билетах`. Лидерство по абсолютному значению вернувшихся клиентов - у самого крупного  оператора `Билеты без проблем`.

# ---
# 
# **Задача 4.1.3.** Опираясь на выводы из задач выше, проверьте продуктовые гипотезы:
# 
# - **Гипотеза 1.** Тип мероприятия влияет на вероятность возврата на Яндекс Афишу: пользователи, которые совершили первый заказ на спортивные мероприятия, совершают повторный заказ чаще, чем пользователи, оформившие свой первый заказ на концерты.
# - **Гипотеза 2.** В регионах, где больше всего пользователей посещают мероприятия, выше доля повторных заказов, чем в менее активных регионах.
# 
# ---

# In[59]:


df_fe


# **Гипотеза 1** странно сформулирована. 
# 
# *Тип мероприятия влияет на вероятность возврата на Яндекс Афишу* - да, влияет. Чаще всего повторные заказы делают пользователи, посетившие выставки, театр и концерты. 
# 
# *пользователи, которые совершили первый заказ на спортивные мероприятия, совершают повторный заказ чаще, чем пользователи, оформившие свой первый заказ на концерты* - наоборот: повторную покупку пользователи, совершившие первый заказ на спортмеропрития, совершают реже пользователей, первый заказ которых - концерты. 

# **Гипотеза 2** не поддерждается - это видно по данным одного из предыдущи графиков. У наиболее активных регионов доля хоть и высокая, но существуют регионы с большей доле повторных заказов.

# ---
# 
# #### 4.2. Исследование поведения пользователей через показатели выручки и состава заказа
# 
# Изучите количественные характеристики заказов пользователей, чтобы узнать среднюю выручку сервиса с заказа и количество билетов, которое пользователи обычно покупают.
# 
# Эти метрики важны не только для оценки выручки, но и для оценки вовлечённости пользователей. Возможно, пользователи с более крупными и дорогими заказами более заинтересованы в сервисе и поэтому чаще возвращаются.
# 
# ---
# 
# **Задача 4.2.1.** Проследите связь между средней выручкой сервиса с заказа и повторными заказами.
# 
# - Постройте сравнительные гистограммы распределения средней выручки с билета (`avg_revenue_rub`):
#     - для пользователей, совершивших один заказ;
#     - для вернувшихся пользователей, совершивших 2 и более заказа.
# - Ответьте на вопросы:
#     - В каких диапазонах средней выручки концентрируются пользователи из каждой группы?
#     - Есть ли различия между группами?
# 
# Текст на сером фоне:
#     
# **Рекомендация:**
# 
# 1. Используйте одинаковые интервалы (`bins`) и прозрачность (`alpha`), чтобы визуально сопоставить распределения.
# 2. Задайте параметру `density` значение `True`, чтобы сравнивать форму распределений, даже если число пользователей в группах отличается.
# 
# ---
# 

# **Здесь задача некорректно сформулирована: сказано построить распределение выручки с билета, но в скобках указно `avg_revenue_rub`, хотя это средняя выручка с заказа, в котором может быть несколько билетов. В следующей задаче речь снова идет про выручку с заказа**

# При расчете моды по признаку `is_two` были выявлены нулевые значения исходного поля `revenue` преимущественно у оператора Билеты без проблем. Нулевые цены будут искажать статистику при работе с показателями выручки. Исключим такие записи из датафрейма. Отрицательные выручки - возвраты - не велики и не многочисленны, можно их оставить.

# In[60]:


dff = df.loc[df['avg_revenue_rub'] != 0]
# dff


# In[61]:


#определим стоимость билета и запишем ее в новое поле
dff['avg_ticket_price']=dff['avg_revenue_rub']/dff['avg_tickets_per_order']


# In[62]:


#гистограмма
plt.figure(figsize=(10,7))
sns.histplot(data=dff, x='avg_ticket_price', hue='is_two', bins=100, kde=True, stat='density', alpha=0.5)
plt.fontsize=14
# оранжевый график должен быть для is_two=1. 
#Почему-то задание легенды plt.legend приводит к неожиданной очередности. Поэтому сначала задаем 2+ заказов, потом 1
# так легенда сооветствует автоматически создаваемой
#plt.legend(labels=['2+ заказов','1 заказ'])  
plt.title('Распределение средней цены билета в разных группах пользователей')
plt.xlabel('Средняя цена билета')
plt.ylabel('Плотность')
plt.show()

#гистограмма
plt.figure(figsize=(8,3))
qq=dff.loc[dff['is_two']==0]
sns.histplot(data=qq, x='avg_ticket_price', bins=100, kde=True, stat='count', alpha=0.5)
plt.fontsize=14
plt.title('Распределение средней цены билета среди пользователей, совершивших 1 заказ')
plt.xlabel('Средняя цена билета')
plt.ylabel('Количество')
plt.show()

#гистограмма
plt.figure(figsize=(8,3))
ww=dff.loc[dff['is_two']==1]
sns.histplot(data=ww, x='avg_ticket_price', bins=100, kde=True, stat='count', alpha=0.5)
plt.fontsize=14
plt.title('Распределение средней цены билета среди пользователей с двумя и более заказами')
plt.xlabel('Средняя цена билета')
plt.ylabel('Количество')
plt.show()

plt.figure(figsize=(8,3))
sns.histplot(data=dff, x='avg_ticket_price', hue='is_two', bins=100, kde=True, stat='count', alpha=0.5)
plt.fontsize=14
plt.title('Распределение средней цены билета среди пользователей с двумя и более заказами')
plt.xlabel('Средняя цена билета')
plt.ylabel('Количество')
plt.show()


# In[63]:


# d1=dff['avg_ticket_price'].loc[dff['is_two']==0]
# d2=dff['avg_ticket_price'].loc[dff['is_two']==1]
# kde1 = stats.gaussian_kde(d1)
# kde2 = stats.gaussian_kde(d2)
# x_vals1 = np.linspace(d1.min(), d1.max(), 1000)
# y_vals1 = kde(x_vals1)
# x_vals2 = np.linspace(d2.min(), d2.max(), 1000)
# y_vals2 = kde(x_vals2)
# peak_x1 = x_vals1[np.argmax(y_vals1)]  
# peak_y1 = y_vals1.max()            
# peak_x2 = x_vals2[np.argmax(y_vals2)]  
# peak_y2 = y_vals2.max()            
# print(f"Пик KDE1: x = {peak_x1:.3f}, плотность = {peak_y1:.4f}")
# print(f"Пик KDE2: x = {peak_x2:.3f}, плотность = {peak_y2:.4f}")
# print(d1.count())
# print(d2.count())


# Пользователи, совершивших два и более заказа, чаще покупают более дорогие билеты,

# In[64]:


dff.groupby(['is_two'])['avg_ticket_price'].mean()


# In[ ]:





# In[65]:


aaa = dff.loc[dff['is_two'] == 1, 'avg_ticket_price']
type(aaa)
aaa.count()
aaa.mode()


# ---
# 
# **Задача 4.2.2.** Сравните распределение по средней выручке с заказа в двух группах пользователей:
# 
# - совершившие 2–4 заказа;
# - совершившие 5 и более заказов.
# 
# Ответьте на вопрос: есть ли различия по значению средней выручки с заказа между пользователями этих двух групп?
# 
# ---
# 

# In[66]:


# dff['is_two_four'] = ((dff['total_orders'] > 1) & (dff['total_orders'] < 5)).astype(int)
# dff

dff['is_two_four'] = 0
dff.loc[dff['total_orders'] <= 1, 'is_two_four'] = 0
dff.loc[(dff['total_orders'] >= 2) & (dff['total_orders'] <= 4), 'is_two_four'] = 1
dff.loc[dff['total_orders'] >= 5, 'is_two_four'] = 2


# In[67]:


#гистограмма
plt.figure(figsize=(14,5))
aa=dff[dff['is_two_four'] >= 1]
sns.histplot(data=aa, x='avg_revenue_rub', hue='is_two_four', 
             palette='muted',
             bins=100, kde=True, stat='density', alpha=0.5)
plt.fontsize=14
# plt.legend(labels=['KDE, 5+ заказов','KDE, 2-4 заказа','5+ заказов','2-4 заказа','k']) ###########
plt.title('Распределение средней выручки с заказа')
plt.xlabel('Средняя выручка с заказа')
plt.ylabel('Плотность')
plt.show()

# #гистограмма
# plt.figure(figsize=(10,5))
# aa=dff[dff['is_two_four'] >= 1]
# sns.histplot(data=aa, x='avg_revenue_rub', hue='is_two_four', palette='muted',
#                  bins=100, kde=True, stat='count', alpha=0.5)
# plt.fontsize=14
# plt.title('Распределение средней выручки с заказа')
# plt.xlabel('Средняя выручка с заказа')
# plt.ylabel('Количество')
# plt.show()


# Мода средней выручки с пользователей, совершивших 5 и более заказов, ниже аналогичного значения для пользователей с 2-4 заказами:

# In[68]:


dff.groupby('is_two_four')['avg_revenue_rub'].apply(lambda x: x.mode())


# Большинство пользователей первой группы (2-4 заказа) приносят минимальную среднюю выручку. С ростом выручки число пользователей резко снижается. 
# Характер распределения средней выручки для пользователей 2й группы (5+ заказов) несколько иной: сначала с ростом выручки число пользователей растет, затем после достижения максимальных значений снижается сильнее, чем у пользователей 1 группы. 
# Пользователи из 2й группы чаще приносят бОльшую среднюю выручку.

# In[69]:


dff.groupby(['is_two_four'])['avg_revenue_rub'].mean()


# Ответьте на вопрос: есть ли различия по значению средней выручки с заказа между пользователями этих двух групп?
#     
# Значение средней выручки для покупателей с 5 и более заказами несущественно выше аналогичного значения для группы пользователей с 2-4 заказами - 585 руб. против 582.7 руб. 

# ---
# 
# **Задача 4.2.3.** Проанализируйте влияние среднего количества билетов в заказе на вероятность повторной покупки.
# 
# - Изучите распределение пользователей по среднему количеству билетов в заказе (`avg_tickets_count`) и опишите основные наблюдения.
# - Разделите пользователей на несколько сегментов по среднему количеству билетов в заказе:
#     - от 1 до 2 билетов;
#     - от 2 до 3 билетов;
#     - от 3 до 5 билетов;
#     - от 5 и более билетов.
# - Для каждого сегмента подсчитайте общее число пользователей и долю пользователей, совершивших повторные заказы.
# - Ответьте на вопросы:
#     - Как распределены пользователи по сегментам — равномерно или сконцентрировано?
#     - Есть ли сегменты с аномально высокой или низкой долей повторных покупок?
# 
# ---

# In[70]:


dff['avg_tickets_per_order'].max()


# In[71]:


plt.figure(figsize=(14,5))
sns.histplot(data=dff, x='avg_tickets_per_order', hue='is_two', 
             palette='muted',
             bins=12, kde=False, stat='density', alpha=0.8)
plt.fontsize=14
# plt.legend(labels=['KDE, 5+ заказов','KDE, 2-4 заказа','5+ заказов','2-4 заказа','k']) ###########
plt.title('Распределение среднего количества билетов в заказе')
# plt.xlabel('Количество билетов')
plt.show()


# Пользователи, совершившие 2+ заказа, чаще покупают 2-4 билета. Совершившие 1 заказ - также 2-4 билета, но доля таких ниже. Распределение пользователей с одним заказом болшее равномерное - чаще готовы покупать ~4 билета. Доли покупателей с заказом менее двух билетов примерно одинаковые в обеих группах пользователей.

# Разделим пользователей на несколько сегментов по среднему количеству билетов в заказе:
# - от 1 до 2 билетов - группа A;
# - от 2 до 3 билетов - группа B; 
# - от 3 до 5 билетов - группа C;
# - от 5 и более билетов - группа D.
# 
# В сводной таблице посчитаем:
#  - каждого сегмента общее число пользователей
#  - долю пользователей, совершивших повторные заказы.

# In[72]:


dff['ticket_group'] = pd.cut(dff['avg_tickets_per_order'], bins=[1,2,3,5,100], right=False, labels=['A','B','C','D'])


# In[73]:


#проверка на пропуски
dff.loc[dff['ticket_group'].isna()==True]['avg_tickets_per_order']
dff['ticket_group'].value_counts().sum() #???


# In[74]:


pt_t = pd.pivot_table(data=dff, index='ticket_group', values=['total_orders'], columns='is_two', aggfunc=['count'])
df_t = pd.DataFrame(pt_t)
df_t[('count', 'total_orders', 'total')] = df_t[('count', 'total_orders', 0)]+df_t[('count', 'total_orders', 1)]
df_t[('count', 'total_orders', 'share2')] = df_t[('count', 'total_orders', 1)]/(df_t[('count', 'total_orders', 0)]+df_t[('count', 'total_orders', 1)])
df_t


# Ответьте на вопросы:
# 
# Как распределены пользователи по сегментам — равномерно или сконцентрировано?
# - Пользователи распределены неравномерно. 
# 
# Есть ли сегменты с аномально высокой или низкой долей повторных покупок?
# - Немногим больше 50% пользователей в группах A (1-2 билетов) и С (3-5 билетов) совершают две и более покупки (заказа)
# - ~72% пользователей, купивших 2-3 билета совершают не менее 2 заказов - можно считать показатель аномально высоким.
# - Только 19% пользователей с заказами с более, чем 5 билетов, возвращаются за повторной покупкой.

# In[75]:


df_t[('count', 'total_orders', 'share2')].sort_values(ascending=False).plot(
    kind='bar', 
    xlabel='Группа пользователей', 
    ylabel='Доля', 
    legend=False,
    rot=0, 
    title='Доля пользователей, совершивших не менее двух заказов, в общем числе пользователей', 
    figsize=(10,4),
    fontsize = 12,
    color = ['royalblue'],
    edgecolor='black'
    )
plt.show()


# ---
# 
# #### 4.3. Исследование временных характеристик первого заказа и их влияния на повторные покупки
# 
# Изучите временные параметры, связанные с первым заказом пользователей:
# 
# - день недели первой покупки;
# - время с момента первой покупки — лайфтайм;
# - средний интервал между покупками пользователей с повторными заказами.
# 
# ---
# 
# **Задача 4.3.1.** Проанализируйте, как день недели, в которой была совершена первая покупка, влияет на поведение пользователей.
# 
# - По данным даты первого заказа выделите день недели.
# - Для каждого дня недели подсчитайте общее число пользователей и долю пользователей, совершивших повторные заказы. Результаты визуализируйте.
# - Ответьте на вопрос: влияет ли день недели, в которую совершена первая покупка, на вероятность возврата клиента?
# 
# ---
# 

# In[76]:


# Определяем номер днянедели
dff['weekday'] = dff['first_order_dt'].dt.weekday + 1


# In[77]:


pt_wd = pd.pivot_table(data=dff, index='weekday', values=['total_orders'], columns='is_two', aggfunc=['count'])
df_wd = pd.DataFrame(pt_wd)
df_wd[('count', 'total_orders', 'total')] = df_wd[('count', 'total_orders', 0)]+df_wd[('count', 'total_orders', 1)]
df_wd[('count', 'total_orders', 'share2')] = df_wd[('count', 'total_orders', 1)]/(df_wd[('count', 'total_orders', 0)]+df_wd[('count', 'total_orders', 1)])
df_wd


# In[78]:


df_wd[('count', 'total_orders', 'share2')].sort_values(ascending=False).plot(
    kind='bar', 
    xlabel='День недели', 
    ylabel='Доля', 
    legend=False,
    rot=0, 
    title='Доля пользователей, совершивших не менее двух заказов, в общем числе пользователей', 
    figsize=(10,4),
    fontsize = 12,
    color = ['royalblue'],
    edgecolor='black'
    )
plt.show()


# День недели не влияет на вероятность возврата пользователя

# ---
# 
# **Задача 4.3.2.** Изучите, как средний интервал между заказами влияет на удержание клиентов.
# 
# - Рассчитайте среднее время между заказами для двух групп пользователей:
#     - совершившие 2–4 заказа;
#     - совершившие 5 и более заказов.
# - Исследуйте, как средний интервал между заказами влияет на вероятность повторного заказа, и сделайте выводы.
# 
# ---
# 

# In[79]:


dff['btw_ord_group'] = pd.cut(dff['total_orders'], bins=[1,2,5,100], right=False, labels=['1 заказ','2-4 заказа','5+ заказов'])


# In[80]:


dff['btw_ord_group'].isna().sum()


# In[81]:


# dff


# In[82]:


pt_btw = pd.pivot_table(data=dff, index='btw_ord_group', values=['avg_days_between_orders'], aggfunc=['mean'])
df_btw = pd.DataFrame(pt_btw)
print('Рассчитайте среднее время между заказами для двух групп пользователей:')
df_btw


# In[83]:


dff['avg_days_between_orders'].max()


# In[84]:


plt.figure(figsize=(14,5))
sns.histplot(data=dff, x='avg_days_between_orders', #hue='is_two', 
#              palette='muted',
             bins=15, kde=True, stat='probability', alpha=0.8)
plt.fontsize=14
plt.title('Распределение среднего интервала между заказами')
plt.show()


# Вероятность того, что пользователь сделает повторный заказ снижается по мере увеличения интервала между заказами. Первые 10 дней - наиболее важные: вероятность возврата составляет 50%. К концу второй декады вероятность снижается до 20%.

# In[ ]:





# ---
# 
# #### 4.4. Корреляционный анализ количества покупок и признаков пользователя
# 
# Изучите, какие характеристики первого заказа и профиля пользователя могут быть связаны с числом покупок. Для этого используйте универсальный коэффициент корреляции `phi_k`, который позволяет анализировать как числовые, так и категориальные признаки.
# 
# ---
# 
# **Задача 4.4.1:** Проведите корреляционный анализ:
# - Рассчитайте коэффициент корреляции `phi_k` между признаками профиля пользователя и числом заказов (`total_orders`). При необходимости используйте параметр `interval_cols` для определения интервальных данных.
# - Проанализируйте полученные результаты. Если полученные значения будут близки к нулю, проверьте разброс данных в `total_orders`. Такое возможно, когда в данных преобладает одно значение: в таком случае корреляционный анализ может показать отсутствие связей. Чтобы этого избежать, выделите сегменты пользователей по полю `total_orders`, а затем повторите корреляционный анализ. Выделите такие сегменты:
#     - 1 заказ;
#     - от 2 до 4 заказов;
#     - от 5 и выше.
# - Визуализируйте результат корреляции с помощью тепловой карты.
# - Ответьте на вопрос: какие признаки наиболее связаны с количеством заказов?
# 
# ---

# In[85]:


#содаем новый столбец для сегментации пользователей по total_orders
dff['total_orders_segment'] = pd.cut(dff['total_orders'], bins=[1,2,5,1000], right=False, labels=['A','B','C'])
print('Проверка пропусков в новом поле total_orders_segment:',dff['total_orders_segment'].isna().sum())
#Формируем два датафрейма с ограниченным перечнем полей для сравнения матриц корреляций
df_corr1 = dff[['total_orders','first_device','first_region','first_service','first_event_type','avg_revenue_rub',
                'avg_tickets_per_order','avg_days_between_orders','avg_ticket_price']]
df_corr2 = dff[['total_orders_segment','first_device','first_region','first_service','first_event_type','avg_revenue_rub',
                'avg_tickets_per_order','avg_days_between_orders','avg_ticket_price']]

int_cols1=['total_orders','avg_revenue_rub','avg_tickets_per_order','avg_days_between_orders','avg_ticket_price']
int_cols2=['avg_revenue_rub','avg_tickets_per_order','avg_days_between_orders','avg_ticket_price']
# bins_dict = {'total_orders': [1, 2, 5, 1000]}
corr_matrix1 = df_corr1.phik_matrix(interval_cols=int_cols1)#), bins=bins_dict)
corr_matrix2 = df_corr2.phik_matrix(interval_cols=int_cols2)#), bins=bins_dict)


# In[86]:


#corr_matrix1['total_orders']


# In[87]:


#corr_matrix2['total_orders_segment']


# In[88]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title('Исходный вариант Total orders')
ax2.set_title('Сегментированный Total orders')
sns.heatmap(corr_matrix1[['total_orders']], ax=ax1, cmap='coolwarm', annot=True, fmt='.2f', cbar=False)
sns.heatmap(corr_matrix2[['total_orders_segment']], ax=ax2, cmap='coolwarm', annot=True, fmt='.2f', cbar=True, yticklabels=False)
plt.show()


# Наибольшая связь с количеством заказов прослеживается у полей `"Среднее количество дней между заказами/avg_days_between_orders"` `"Среднее коичество билетов в заказе/avg_tickets_per_order"`. Оценка влияния этих факторов на вероятность возврата приведена в предыдущих пунктах.

# ### 5. Общий вывод и рекомендации
# 
# В конце проекта напишите общий вывод и рекомендации: расскажите заказчику, на что нужно обратить внимание. В выводах кратко укажите:
# 
# - **Информацию о данных**, с которыми вы работали, и то, как они были подготовлены: например, расскажите о фильтрации данных, переводе тенге в рубли, фильтрации выбросов.
# - **Основные результаты анализа.** Например, укажите:
#     - Сколько пользователей в выборке? Как распределены пользователи по числу заказов? Какие ещё статистические показатели вы подсчитали важным во время изучения данных?
#     - Какие признаки первого заказа связаны с возвратом пользователей?
#     - Как связаны средняя выручка и количество билетов в заказе с вероятностью повторных покупок?
#     - Какие временные характеристики влияют на удержание (день недели, интервалы между покупками)?
#     - Какие характеристики первого заказа и профиля пользователя могут быть связаны с числом покупок согласно результатам корреляционного анализа?
# - Дополните выводы информацией, которая покажется вам важной и интересной. Следите за общим объёмом выводов — они должны быть компактными и ёмкими.
# 
# В конце предложите заказчику рекомендации о том, как именно действовать в его ситуации. Например, укажите, на какие сегменты пользователей стоит обратить внимание в первую очередь, а какие нуждаются в дополнительных маркетинговых усилиях.

# В соответствии с заданием с помощью SQL на основе данных таблиц БД Афиша был собран датасет с данными о заказах Афиши. Количество строк - ~291 тыс. Для конвертации курса казахской тенге в рубли использовался датасет (357 строк), содержащий выгрузку данных о динамике курса валюты по данным сайта ЦБ. Датасеты были объединены в один, выручка в тенге конвертирована в рубли. Из объединенного датасета были исключены строки со слишком высокими значениями выручки. Фильтрация по 99 перцинтилю позволила снизить искажения статистики и сделало более равномерным распредедение заказов по типам меропритий.
# 
# После предварительной обработки был сформирован новый датасет - профиль пользователя, содержащий агргированные данные по каждому пользователю. Новый датасет также был отфильтрован по 95% процентилю - для исключения из анализа пользователей с аномально большими заказами. 
# Т.к. для части пользователей значения в поле `выручка/avg_revenue_rub` равнялись нулю, то для анализа показателей, связанных с выручкой соответствующие строки также были отфильтрованы. 
# 
# Для целей проведения аналитических расчетов в ходе выполнения проекта были созданы дополнительные поля, позволяющие категоризировать данные о пользователях в различных разрезах. Основое поле из дополнительных - `is_two`, содержащее указание на то, что пользователь совершал более одного заказа. Наличие в этом поле значения, равного единице, указаывает, что пользователь вернулся в сервис. 

# *Сколько пользователей в выборке? Как распределены пользователи по числу заказов? Какие ещё статистические показатели вы подсчитали важным во время изучения данных?*
#     
#     Обновленные данные после фильтрации по 95 процентилю:
# - Общее число пользователей в выборке:  20864. Пользователи по числу заказов распределены неравномерно. С ростом числа заказов, количество пользователей экспоненциально снижается. 
# - Средняя выручка с одного заказа:  581.11 руб.
# - 60% пользователей вернулись. При этом 26% совершили 5 и более заказов.
# - Пользователи чаще всего совершали 2 заказа, заказывая в среднем чуть менее 3 билетов. Медианное значение периода между заказами - 9 дней. При этом среднее - 17, т.к. в выборке есть пользователи с длительными периодом между заказами (несколько месяцев)
# 
# *Какие признаки первого заказа связаны с возвратом пользователей?*
#     
#     С возвратом пользователей связаны количество билетов в заказе, тип мероприятия, устройство, регион и оператор.
# - Чаще всего возвращаются пользователи, посетившие концерты, театры и "другое" (вероятно, что эту категорию можно детализировать).
# - Доля вернувшихся пользователей различается по регионам и операторам. 
# - Пользователей, совершающих первый заказ с компьютера меньше, чем пользователей, совершающих заказы с мобильных устройств. Но возвращаются desktop-пользователи немного чаще. 
# 
# *Как связаны средняя выручка и количество билетов в заказе с вероятностью повторных покупок?*
#     
# - С ростом средней выручки количество пользователей ожидаемо снижается. Но характер распределения разный у разных групп пользователей: большинство пользователей, сделавших только 1 заказ, приносят минимальную выручку, а далее с ростом выручки количество пользователей экспоненциально снижается. 
# - В группе пользователей с 2 и более заказами наблюдается несколько иная динамика: минимальную среднюю выручку формирует небольшое количество пользователей. Далее, с ростом выручки количество пользователей увеличивается и, достигнув пика, начинается снижаться аналогично динамике первой группы. Вероятно пользователи, сделавшие несколько заказов, имеюют более высокие доходы, т.к. предпочитают посещение более дорогих мероприятий. 
# 
#  - Чаще всего возвращаются пользователи, покупающие 2-3 билета (72% вернувшихся). Второе место - у группы покупателей 3-5 билетов (54%), третье - 1-2 билета (51%), четвертое - 5+ билетов (19%) (вероятно группа сама по себе не многочисленна)
# 
# *Какие временные характеристики влияют на удержание (день недели, интервалы между покупками)?*
# 
#     День недели практически не влияет на удержание: доля вернувшихся пользователей колеблется около 60% +/-2%. 
#     
#     Интервалы между покупками влияют существенно: вероятность экспоненциально снижается с ростом инетрвала. Наиболее критичными являются первые 10 дней: вероятность возврата составляет 50%. За следующие 10 дней вероятность снижается до 20% и т.д.
# 
# *Какие характеристики первого заказа и профиля пользователя могут быть связаны с числом покупок согласно результатам корреляционного анализа?*
# 
#     Корреляционный анализ показал, что наибольшую связь с `количеством заказов` имеют признаки `среднее количество дней между заказами` и `среднее количество билетов в заказе`.

# **Рекомендации**
# 
# Общий принцип рекомендаций: Максимизировать эффект сильных качеств, улучшить показатели по отстающим направлениям.
#     Необходимо работать над улучшением качества мероприятий, доступных на платформе. Интерфейс и логика работы приложения должна быть доработана таким образом, чтобы чаще вызывать желание пользователя сделать повторный заказ. При этом важно не допускать излишней навязчивости. 
# 
# 1. Необходимо провести детальный анализ регионов с меньшей долей возвратов пользователей. Возможно, что качество мероприятий, доступных на платформе оставляет желать лучшего. 
# 2. Аналогичная рекомендация по операторам: должна быть разработана/улучшена система работы с операторами, введены специальные стимулирующие меры и т.д.
# 3. Анализ показал, что возвращаются пользователи, предположительно, с большим доходом (выше средняя вырычка - больше покупают более дорогие мероприятия). Вероятно, нужно дополнительно оценить необходимость концентрации на мероприятиях более высокой ценовой категории. Точнее, на - расширении перечня таких мероприятий. 
# 4. При выявлении соответствуюещей необходимости проводить модернизацию интерфейса приложения/сайта.
# 5. Т.к. количество дней сильно влияет на возврат пользователей, необходимо провести дополнительный анализ: нужно определить, что конкретно заставляет пользователя вернуться за следующим заказом в разрезе категорий мероприятий. По итогам провести соотвествующую доработку приложения и модернизировать алгоритмы отбора мероприятий. 
# 6. Между значимыми изменениями в работе проекта необходимо проводить A/B тестирование, внедрять изменения поэтапно для разных групп пользователей. 
# 7. 
# 
# 

# ### 6. Финализация проекта и публикация в Git
# 
# Когда вы закончите анализировать данные, оформите проект, а затем опубликуйте его.
# 
# Выполните следующие действия:
# 
# 1. Создайте файл `.gitignore`. Добавьте в него все временные и чувствительные файлы, которые не должны попасть в репозиторий.
# 2. Сформируйте файл `requirements.txt`. Зафиксируйте все библиотеки, которые вы использовали в проекте.
# 3. Вынести все чувствительные данные (параметры подключения к базе) в `.env`файл.
# 4. Проверьте, что проект запускается и воспроизводим.
# 5. Загрузите проект в публичный репозиторий — например, на GitHub. Убедитесь, что все нужные файлы находятся в репозитории, исключая те, что в `.gitignore`. Ссылка на репозиторий понадобится для отправки проекта на проверку. Вставьте её в шаблон проекта в тетрадке Jupyter Notebook перед отправкой проекта на ревью.

# https://github.com/dimaxi1007/afisha
