# Network Traffic Prediction Using Gated Recurrent Units (GRU)

An end-to-end deep learning framework designed to model, clean, and forecast network traffic volume using synthetic time-series sequences and optimized Gated Recurrent Unit (GRU) networks.

## 📌 Project Architecture & Workflow
This repository implements a complete machine learning pipeline divided into four main operational phases:
[Raw Synthetic Data] ➔ [Preprocessing Pipeline] ➔ [Sliding Window Generator] ➔ [GRU Model] ➔ [Evaluation]
### 1. Data Generation & Characteristics (`Dataset.csv`)
The project utilizes a structured synthetic dataset containing **94,272 observations** and **7 attributes** that mimic real-world network packet Captures (e.g., Wireshark outputs):
* **No:** Sequential row identifier.
* **Time:** Continuous chronological elapsed time. Imbued with micro-level randomized negative fluctuations (e.g., `-0.005`) to simulate real-world logging noise.
* **Source & Destination:** Textual, synthetically distributed IP addresses representing communication endpoints.
* **Protocol:** Categorical string logs of protocol types (`TCP`, `UDP`, `TLSv1.2`, `HTTP`, `SSL`, `QUIC`).
* **Length (Target Feature):** Positive integer byte counts generated via overlapping mathematical sine/cosine waves to simulate bursty, periodic network traffic spikes.
* **Info:** Randomized alphanumeric technical details and packet flags.

### 2. Automated Preprocessing Pipeline (`database.py`)
To make the raw logs digestible by a deep learning architecture, an automated cleansing pipeline executes the following:
* **Feature Pruning:** Removes metadata and complex text attributes (`No`, `Source`, `Destination`, `Info`) that introduce overhead without statistical value.
* **Categorical Encoding:** Converts categorical strings into integers using `LabelEncoder`.
* **Feature Normalization:** Maps all variables (`Time`, `Protocol`, `Length`) into a unified bounding interval of **`[0, 1]`** via `MinMaxScaler` to guarantee smooth gradient descents.
* **Result:** Outputs a sanitized, purely numerical matrix saved inside `cleaned_dataset_sample.csv`.

### 3. Temporal Sequencing (Sliding Window Strategy)
Since standard deep networks do not naturally comprehend order, the continuous cleaned series is transformed into overlapping sequences:
* **Look-Back Window (`LOOK_BACK = 128`):** For any step $t$, the model takes the historical context of the past 128 steps across all 3 features.
* **Input Tensor Shape:** Reshapes the series into a 3D matrix layout of `(Samples, 128, 3)` where `3` represents the synchronized attributes (`Time`, `Protocol`, `Length`).

### 4. Deep Learning Model & Evaluation (`main.py`)
* **Core Architecture:** Built utilizing **Gated Recurrent Units (GRU)**. GRU’s dual-gate mechanism (Reset and Update) efficiently captures long-term dependencies in the traffic sequence while using fewer parameters than traditional LSTMs, leading to significantly faster inference rates.
* **Training & Splitting:** Data split into Train (70%), Validation (10%), and Test (20%) sets. 
* **Deliverables:** The training produces precise convergence logs and saves performance graphs (`final_results.jpg`, `target_column_analysis.png`), validating a near-perfect alignment between the ground-truth traffic length and the GRU's real-time forecasted outputs.
  # پیش‌بینی ترافیک شبکه با استفاده از واحدهای بازگشتی گیت‌دار (GRU)

یک چارچوب یادگیری عمیق گام‌به‌گام برای مهندسی داده، پاک‌سازی و پیش‌بینی پویای حجم ترافیک شبکه با استفاده از سری‌های زمانی شبیه‌سازی‌شده و معماری بهینه شبکه عصبی GRU.

## 📌 معماری و مراحل اجرای پروژه
این پروژه شامل یک خط لوله (Pipeline) کامل یادگیری ماشین است که فرآیند خود را در ۴ فاز اصلی جلو می‌برد:
### ۱. ساختار داده‌های اولیه (`Dataset.csv`)
پروژه از یک مجموعه داده ساختاریافته دستی شامل **۹۴,۲۷۲ سطر** و **۷ ستون** استفاده می‌کند که رفتار بسته‌های واقعی شبکه را بازسازی می‌کند:
* **No:** شناسه عددی صعودی سطرها.
* **Time:** زمان اعشاری صعودی. حاوی مقادیر منفی بسیار کوچک و رندوم (مانند `-0.005`) جهت شبیه‌سازی نویزهای زمانی شبکه‌های واقعی.
* **Source و Destination:** آدرس‌های متنی IP مبدأ و مقصد که به صورت تصادفی توزیع شده‌اند.
* **Protocol:** پروتکل‌های شبکه متنی شامل (`TCP`, `UDP`, `TLSv1.2`, `HTTP`, `SSL`, `QUIC`).
* **Length (متغیر هدف):** اعداد صحیح مثبت نشان‌دهنده حجم بسته بر حسب بایت که با ترکیب توابع ریاضی سینوسی/کسینوسی الگوبرداری شده تا جهش‌ها و نوسانات متناوب ترافیک را بازسازی کند.
* **Info:** اطلاعات فنی و پرچم‌های سیستمی رندوم.

### ۲. خط لوله پیش‌پردازش هوشمند داده‌ها (`database.py`)
برای تبدیل داده‌های خام به فرمت قابل فهم برای شبکه عصبی، عملیات زیر به صورت خودکار اعمال می‌شود:
* **حذف ویژگی‌های زائد:** ستون‌های غیرضروری آماری (`No`, `Source`, `Destination`, `Info`) حذف می‌شوند.
* **کدگذاری متغیرهای متنی:** تبدیل پروتکل‌های متنی به مقادیر عددی با متد `LabelEncoder`.
* **نرمال‌سازی ابعادی:** انتقال تمامی مقادیر هر ۳ ستون باقی‌مانده به بازه استاندارد **`[0, 1]`** با الگوریتم `MinMaxScaler` جهت پایداری فرآیند آموزش.
* **خروجی:** یک ماتریس تمام‌عددی خالص که در فایل `cleaned_dataset_sample.csv` ذخیره می‌شود.

### ۳. مکانیزم پنجره لغزان زمانی (Temporal Sequencing)
از آنجا که مدل‌های بازگشتی نیاز به درک گذشته دارند، سری زمانی پیوسته به توالی‌های متداخل شکسته می‌شود:
* **طول پنجره (`LOOK_BACK = 128`):** برای پیش‌بینی وضعیت ترافیک در گام $t$، وضعیت ۱۲۸ گام زمانی ماقبل آن از هر ۳ ویژگی به عنوان تاریخچه بررسی می‌شود.
* **ابعاد نهایی ورودی:** تبدیل دیتاست به یک ماتریس سه‌بعدی به فرمت `(Samples, 128, 3)` تا مستقیماً با لایه‌های ورودی شبکه عصبی سازگار باشد.

### ۴. معماری مدل یادگیری عمیق و ارزیابی (`main.py`)
* **ساختار مدل:** پیاده‌سازی‌شده با شبکه‌های **GRU**. گیت‌های داخلی این مدل (Reset و Update) علاوه بر یادگیری الگوهای پیچیده طولانی‌مدت، به دلیل ساختار بهینه‌تر نسبت به LSTM سرعت محاسباتی و استنتاج آنی (Real-time) بسیار بالاتری ارائه می‌دهند.
* **فرآیند آموزش:** تقسیم داده‌ها به بخش‌های آموزش (۷۰٪)، اعتبارسنجی (۱۰٪) و تست (۲۰٪).
* **خروجی و دستاوردها:** ذخیره‌سازی نمودارهای عملکرد نهایی مدل (`final_results.jpg` و `target_column_analysis.png`) که نشان‌دهنده انطباق بسیار دقیق و خطای نزدیک به صفر میان رفتار ترافیک واقعی و پیش‌بینی‌های مدل GRU است.
