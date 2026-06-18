# Network Traffic Prediction Using Gated Recurrent Units (GRU)

An end-to-end deep learning project focused on modeling and forecasting network traffic dynamics using Synthetic Time-Series Data and Gated Recurrent Units (GRU) implemented in PyTorch/Tensorflow.

## Project Overview
This project targets active management of network resources and prevention of congestion by forecasting traffic metrics ahead of time. Leveraging a custom automated data preprocessing pipeline, raw multi-dimensional captured sequences are dynamically transformed into optimized, low-noise series ideal for recurrent structures.

## Dataset Structure

### 1. Raw Dataset (`Dataset.csv`)
The initial unprocessed synthetic archive mimics network packet monitoring utilities, totaling **94,272 rows** and **7 columns**:
* **No:** Sequential observation identifier.
* **Time:** Continuous elapsed time starting from `0.0` seconds. Imbued with micro-level randomized negative jitters (e.g., `-0.005`) to stress-test the model's robustness against chronological anomalies.
* **Source & Destination:** Textual, synthetically generated IP addresses reflecting endpoint locations distributed stochastically.
* **Protocol:** Categorical string descriptors of transport and application layer types (TCP, UDP, TLSv1.2, HTTP, SSL, QUIC) distributed with structured operational frequencies.
* **Length:** Positive integer representations of packet sizing in bytes, synthesized using non-linear math oscillations to model periodic traffic surges.
* **Info:** Randomized alphanumeric logs and flags detailing transmission data.

### 2. Preprocessed Dataset (`cleaned_dataset_sample.csv`)
Through the pipeline, all irrelevant context and noisy sequences are stripped out, yielding an optimized configuration with **3 pure numerical columns**:
* **Time / Protocol / Length:** All values mapped to the unified bounding interval of **`[0, 1]`** via `MinMaxScaler`. Protocol strings are dynamically converted using `LabelEncoder`.

## Model Pipeline & Windowing
* **Look-Back Mechanism:** A rolling window size of `LOOK_BACK = 128` steps translates the temporal logs into a 3D matrix shape of `(Samples, 128, 3)`.
* **Architecture:** Utilizes Gated Recurrent Units (GRU) for reduced computational overhead and faster inference rates compared to traditional LSTM models, making it ideal for real-time tracking.

---
# پیش‌بینی ترافیک شبکه با استفاده از واحدهای بازگشتی گیت‌دار (GRU)

این پروژه یک پیاده‌سازی جامع از مدل‌های یادگیری عمیق به منظور مدل‌سازی و پیش‌بینی پویای ترافیک شبکه است که با بهره‌گیری از داده‌های شبیه‌سازی‌شده زمانی و معماری عمیق GRU توسعه یافته است.

## خلاصه پروژه
هدف اصلی این پژوهش، پیش‌بینی هوشمند بار ترافیکی شبکه برای مدیریت منابع پیشگیرانه و جلوگیری از اشباع پهنای باند است. پروژه مجهز به یک خط لوله پردازشی خودکار است که داده‌های خام را پاک‌سازی، کدگذاری و آماده تزریق به شبکه عصبی بازگشتی می‌کند.

## ساختار دیتابیس و ویژگی‌ها

### ۱. دیتاست دست‌نخورده اولیه (`Dataset.csv`)
نسخه خام شبیه‌سازی‌شده در مجموع شامل **۹۴,۲۷۲ سطر** و **۷ ستون** ساختاریافته است:
* **No (شماره ردیف):** شناسه ساده و صعودی رکوردها.
* **Time (زمان):** اعداد اعشاری صعودی شروع از `0.0` ثانیه. حاوی مقادیر منفی بسیار کوچک و رندوم (مثل `-0.005`) جهت به چالش کشیدن پایداری زمانی مدل.
* **Source و Destination:** آدرس‌های متنی IP مبدأ و مقصد که به صورت تصادفی توزیع شده‌اند.
* **Protocol (پروتکل):** متغیرهای کیفی لایه‌های شبکه (TCP, UDP, TLSv1.2, HTTP, SSL, QUIC) با فراوانی‌های شبیه‌سازی‌شده متعارف.
* **Length (طول بسته):** متغیر کلیدی ترافیک بر حسب بایت با ماهیت عددی صحیح که از توابع نوسانی ریاضی برای بازسازی امواج ترافیکی شبکه واقعی الگوبرداری کرده است.
* **Info:** اطلاعات متنی و پرچم‌های سیستمی تصادفی.

### ۲. دیتاست پالایش‌شده (`cleaned_dataset_sample.csv`)
ستون‌های غیرضروری و نویزهای ساختاری حذف شده و حاصل آن یک ماتریس بهینه با **۳ ستون خالص عددی** است:
* **Time / Protocol / Length:** تمامی مقادیر بدون استثنا با الگوریتم `MinMaxScaler` به بازه استاندارد **`[0, 1]`** منتقل شده‌اند و پروتکل‌ها با `LabelEncoder` مقدار عددی گرفته‌اند.

## نحوه ایجاد توالی و معماری مدل
* **مکانیزم پنجره لغزان:** با در نظر گرفتن پنجره زمانی تاریخچه (`LOOK_BACK = 128`)، مدل ۱۲۸ گام زمانی گذشته را برای پیش‌بینی وضعیت گام بعدی تحلیل می‌کند و ساختار داده‌ها را به ابعاد سه‌بعدی `(Samples, 128, 3)` می‌برد.
* **انتخاب مدل:** ساختار GRU به دلیل داشتن پارامترهای بهینه محاسباتی نسبت به LSTM، سرعت بسیار بالاتری در آموزش و استنتاج آنی (Real-time) ارائه می‌دهد.
