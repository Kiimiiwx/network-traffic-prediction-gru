import os
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

from database import auto_preprocess_csv
from feature_analyzer import FeatureAnalyzer

# تنظیمات
MODEL_PATH = "final_gru_traffic_model.keras"
HASH_PATH = "last_dataset.hash"
LOOK_BACK = 128
EPOCHS_FINE_TUNE = 20
EPOCHS_NEW_MODEL = 100
BATCH_SIZE = 128
TEST_SIZE = 0.2
VAL_SIZE = 0.1


def get_dataset_hash_sha256(file_path: str) -> str:
    """
    تولید اثر انگشت (هش) SHA-256 از فایل CSV.

    این تابع برای ایجاد یک شناسه منحصر به فرد از محتوای فایل CSV استفاده می‌شود.
    اگر محتوای فایل تغییر کند، مقدار هش نیز تغییر می‌کند.

    دلیل استفاده:
    - امکان بررسی تغییرات در داده‌های ورودی
    - اطمینان از آموزش مدل با نسخه مشخصی از داده‌ها
    - جلوگیری از آموزش‌های تکراری روی داده‌های تغییرناکرده

    پارامترها:
    -----------
    file_path : str
        مسیر فایل CSV

    بازگشت:
    -------
    str
        هش SHA-256 به صورت رشته ۶۴ کاراکتری هگزادسیمال
    """
    # بارگذاری دیتافریم از فایل CSV
    # دلیل: برای محاسبه هش نیاز به محتوای کامل فایل داریم
    df = pd.read_csv(file_path)

    # تبدیل دیتافریم به رشته CSV (بدون index)
    # دلیل: to_csv همه داده‌ها (شامل هدرها) را به صورت رشته برمی‌گرداند
    # استفاده از encoding utf-8 برای پشتیبانی از کاراکترهای فارسی
    data_bytes = df.to_csv(index=False).encode("utf-8")

    # محاسبه هش SHA-256
    # دلیل: SHA-256 امن‌تر و قابل اطمینان‌تر از الگوریتم‌های قدیمی مثل MD5 است
    # طول هش: ۶۴ کاراکتر هگزادسیمال (۲۵۶ بیت)
    return hashlib.sha256(data_bytes).hexdigest()


def is_new_dataset_sha256(file_path: str, hash_file: str = HASH_PATH) -> bool:
    """
    بررسی می‌کند که آیا فایل داده جدید است یا خیر.

    این تابع با مقایسه هش فعلی فایل با هش ذخیره‌شده قبلی،
    تغییرات در داده‌ها را شناسایی می‌کند.

    منطق کار:
    ۱. اگر فایل هش وجود نداشته باشد → داده جدید است (ذخیره هش جدید)
    ۲. اگر فایل هش وجود داشته باشد → مقایسه هش‌ها:
       - اگر هش‌ها یکسان باشند → داده تغییر نکرده
       - اگر هش‌ها متفاوت باشند → داده جدید است (به‌روزرسانی هش)

    پارامترها:
    -----------
    file_path : str
        مسیر فایل CSV داده‌ها
    hash_file : str, optional
        مسیر فایل ذخیره هش (پیش‌فرض: HASH_PATH)

    بازگشت:
    -------
    bool
        True: داده جدید است (نیاز به آموزش/بازآموزی مدل)
        False: داده تغییر نکرده (نیازی به آموزش مجدد نیست)
    """
    # محاسبه هش فعلی فایل CSV
    # دلیل: برای مقایسه با هش ذخیره‌شده قبلی
    new_hash = get_dataset_hash_sha256(file_path)

    # بررسی وجود فایل هش قبلی
    if not os.path.exists(hash_file):
        # اگر فایل هش وجود نداشت
        # دلیل: اولین بار است که داده‌ها پردازش می‌شوند
        print(f"Create new HashFile with SHA-256: {hash_file}")

        # ایجاد فایل هش و ذخیره هش جدید
        # دلیل: برای استفاده در بررسی‌های بعدی
        with open(hash_file, "w") as f:
            # ذخیره با پیشوند "SHA256:" برای شفافیت نوع هش
            f.write(f"SHA256:{new_hash}")

        # بازگشت True زیرا داده جدید است
        # دلیل: اولین بار است که این داده‌ها دیده می‌شوند
        return True

    # اگر فایل هش وجود داشت، خواندن هش ذخیره‌شده
    with open(hash_file, "r") as f:
        content = f.read().strip()

    # استخراج هش قدیمی با توجه به فرمت ذخیره‌سازی
    # دلیل: فایل هش ممکن است از نسخه‌های مختلف باشد
    if content.startswith("SHA256:"):
        # اگر هش با پیشوند SHA256: ذخیره شده باشد
        # حذف ۷ کاراکتر اول ("SHA256:") برای گرفتن هش خالص
        old_hash = content[7:]
    else:
        # اگر هش بدون پیشوند ذخیره شده باشد (نسخه‌های قدیمی)
        # دلیل: سازگاری با فایل‌های هش قدیمی‌تر
        old_hash = content

    # مقایسه هش جدید با هش قدیمی
    if new_hash == old_hash:
        # اگر هش‌ها یکسان باشند
        # دلیل: داده‌ها تغییر نکرده‌اند
        print(f"No change in {file_path}")

        # بازگشت False زیرا داده تغییر نکرده
        # دلیل: نیازی به آموزش مجدد مدل نیست
        return False
    else:
        # اگر هش‌ها متفاوت باشند
        # دلیل: داده‌ها تغییر کرده‌اند یا به‌روزرسانی شده‌اند
        print(f"Change in {file_path}")

        # نمایش ۱۶ کاراکتر اول هر هش برای مقایسه بصری
        # دلیل: هش کامل ۶۴ کاراکتری است، نمایش بخشی از آن کافی است
        print(f"  Old Hash: {old_hash[:16]}...")
        print(f"  New Hash:  {new_hash[:16]}...")

        # به‌روزرسانی فایل هش با هش جدید
        # دلیل: برای استفاده در بررسی‌های بعدی
        with open(hash_file, "w") as f:
            f.write(f"SHA256:{new_hash}")

        # بازگشت True زیرا داده جدید است
        # دلیل: داده‌ها تغییر کرده و نیاز به پردازش مجدد دارند
        return True


def build_final_model(input_shape):
    """
    ساخت مدل GRU دو لایه ساده شده برای پیش‌بینی ترافیک شبکه.

    این تابع یک مدل GRU سبک‌تر ایجاد می‌کند که برای سناریوهایی مناسب است
    که داده‌ها محدود هستند یا نیاز به مدل سریع‌تر داریم. ساختار ساده‌شده
    شامل دو لایه GRU با اندازه‌های مختلف و سه لایه Dense می‌باشد.

    ساختار مدل:
    -------------
    ۱. GRU(128) → Dropout(0.2)
    ۲. GRU(64) → Dropout(0.2)
    ۳. Dense(64) → ReLU
    ۴. Dense(32) → ReLU
    ۵. Dense(1)

    پارامترها:
    -----------
    input_shape : tuple
        شکل ورودی مدل به فرم (timesteps, features).
        مثال: (60, 1) برای ۶۰ گام زمانی با ۱ ویژگی

    بازگشت:
    -------
    tensorflow.keras.Sequential
        مدل کامپایل‌شده GRU آماده برای آموزش

    منطق طراحی معماری ساده‌شده:
    ---------------------------
    - لایه اول GRU با ۱۲۸ نورون: استخراج ویژگی‌های زمانی با ظرفیت بالا
    - لایه دوم GRU با ۶۴ نورون: فشرده‌سازی اطلاعات و کاهش ابعاد
    - لایه‌های Dense کم‌تعداد: جلوگیری از overfitting و حفظ سادگی مدل
    - حذف لایه‌های اضافی: برای سرعت بیشتر آموزش و جلوگیری از overfitting
      در داده‌های کوچک‌تر

    مزایای مدل ساده‌شده:
    --------------------
    - آموزش سریع‌تر
    - نیاز به داده کمتر
    - کاهش خطر overfitting
    - تفسیرپذیری بهتر

    نکات:
    -----
    - این مدل برای داده‌های با حجم متوسط مناسب است
    - در صورت داشتن داده زیاد، می‌توان مدل را پیچیده‌تر کرد
    - تنظیم نرخ یادگیری ۰.۰۰۱ برای تعادل سرعت و دقت
    """

    # مرحله ۱: ایجاد مدل Sequential
    # دلیل: استفاده از Sequential برای ساختار خطی و ساده
    model = Sequential([

        # تعریف شکل ورودی
        # دلیل: مشخص کردن ابعاد داده‌های ورودی برای لایه اول
        Input(shape=input_shape),

        # لایه اول GRU: استخراج ویژگی‌های زمانی با حفظ توالی
        # دلیل:
        # - ۱۲۸ نورون برای ظرفیت یادگیری بالا در مرحله اول
        # - return_sequences=True برای ارسال کامل توالی به لایه بعد
        # - نام‌گذاری برای ردیابی بهتر در آموزش
        GRU(128, return_sequences=True, name='gru_1'),

        # Dropout اول: جلوگیری از overfitting
        # دلیل:
        # - ۲۰٪ dropout برای ایجاد تعادل بین یادگیری و تعمیم
        # - تصادفی‌سازی فعال‌سازی‌ها برای افزایش استحکام مدل
        Dropout(0.2),

        # لایه دوم GRU: فشرده‌سازی اطلاعات زمانی
        # دلیل:
        # - ۶۴ نورون (کاهش از ۱۲۸) برای فشرده‌سازی ویژگی‌ها
        # - return_sequences=False برای تبدیل توالی به بردار ثابت
        # - آماده‌سازی ویژگی‌ها برای لایه‌های Dense
        GRU(64, return_sequences=False, name='gru_2'),

        # Dropout دوم: منظم‌سازی نهایی
        # دلیل:
        # - کاهش وابستگی به نورون‌های خاص
        # - افزایش قدرت تعمیم مدل
        Dropout(0.2),

        # لایه‌های کاملاً متصل: تبدیل ویژگی‌های زمانی به پیش‌بینی
        # دلیل کاهش تعداد لایه‌های Dense:
        # - جلوگیری از حذف اطلاعات در داده‌های محدود
        # - کاهش پارامترها و جلوگیری از overfitting
        # - حفظ سادگی و سرعت مدل

        # لایه Dense اول: کاهش ابعاد به ۳۲ نورون
        # دلیل:
        # - تبدیل برزر ۶۴ بعدی از GRU به ۳۲ بعد
        # - استخراج ارتباط‌های غیرخطی با ابعاد مناسب
        Dense(64, activation='relu'),

        # لایه Dense دوم: کاهش بیشتر به ۱۶ نورون
        # دلیل:
        # - ادامه فرآیند استخراج ویژگی‌های مرتبط
        # - آماده‌سازی برای لایه خروجی
        Dense(32, activation='relu'),

        # لایه خروجی: پیش‌بینی مقدار ترافیک
        # دلیل:
        # - ۱ نورون برای خروجی رگرسیون
        # - بدون تابع فعال‌سازی برای خروجی پیوسته
        # - نام 'output' برای شناسایی آسان
        Dense(1, activation='linear', name='output')
    ])

    # مرحله ۲: کامپایل مدل
    # دلیل: تنظیم بهینه‌ساز، تابع خطا و معیارهای ارزیابی
    model.compile(

        # انتخاب بهینه‌ساز Adam
        # دلیل: عملکرد عالی در مسائل رگرسیون زمانی
        optimizer=tf.keras.optimizers.Adam(

            # نرخ یادگیری: ۰.۰۰۱
            # دلیل: مقدار استاندارد برای مدل‌های GRU
            learning_rate=0.001,

            # بتا ۱: ۰.۹ (ضریب مومنتوم)
            # دلیل: نرم کردن به‌روزرسانی پارامترها
            # beta_1=0.9,

            # بتا ۲: ۰.۹۹۹ (ضریب مقیاس‌گذاری)
            # دلیل: تطبیق نرخ یادگیری برای پارامترهای مختلف
            # beta_2=0.999
        ),

        # تابع خطا: Mean Squared Error
        # دلیل: مناسب برای پیش‌بینی مقادیر پیوسته ترافیک
        loss='mse',

        # معیارهای ارزیابی: MAE و MSE
        # دلیل:
        # - MAE: میانگین قدر مطلق خطا (قابل درک‌تر)
        # - MSE: میانگین مربع خطا (همان loss برای مانیتورینگ)
        metrics=['mae', 'mse']
    )

    # مرحله ۳: نمایش خلاصه مدل
    # دلیل: بررسی ساختار نهایی و تعداد پارامترها
    print("GRU Model : Done")
    model.summary()

    # مرحله ۴: بازگشت مدل کامپایل‌شده
    # دلیل: مدل آماده برای آموزش با داده‌های آماده‌شده
    return model


def create_simple_sequences(df: pd.DataFrame, target_col: str, look_back: int):
    """
    ایجاد دنباله‌های زمانی چندمتغیره (Multivariate) برای آموزش مدل‌های GRU.

    این تابع تمام ویژگی‌های موجود در دیتافریم را به عنوان ورودی (X) در نظر می‌گیرد
    تا مدل بتواند علاوه بر روند تاریخی، از سایر فاکتورها (مثل ساعت، روز و ...)
    برای پیش‌بینی دقیق‌تر استفاده کند.

    پارامترها:
    -----------
    df : pd.DataFrame
        دیتافریم پیش‌پردازش شده شامل تمام ستون‌های عددی.
    target_col : str
        نام ستون هدف (مثلاً website_visits).
    look_back : int
        تعداد گام‌های زمانی قبلی (Window Size).

    بازگشت:
    -------
    tuple (X, y)
        - X: آرایه سه‌بعدی (samples, look_back, num_features)
        - y: آرایه یک‌بعدی شامل مقدار هدف بعدی
    """
    # بررسی وجود ستون هدف در دیتافریم
    # اگر این بررسی نباشد، در خط (366)df.columns.get_loc(target_col) خطای KeyError می‌خوریم
    # و برنامه متوقف می‌شود. این بررسی پیام خطای واضح‌تری به کاربر می‌دهد.
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # بررسی اینکه look_back از طول دیتافریم بزرگتر نباشد
    # برای ساخت یک پنجره کامل، به حداقل look_back+1 ردیف نیاز داریم.
    # اگر look_back >= len(df)، هیچ پنجره کاملی وجود نخواهد داشت.
    # در این صورت X و y خالی می‌مانند و مدل داده‌ای برای آموزش نخواهد داشت.
    if look_back >= len(df):
        raise ValueError(f"look_back ({look_back}) >= dataset length ({len(df)})")

    # مرحله ۱: تبدیل کل دیتافریم به آرایه numpy
    # دلیل: استفاده از تمام ستون‌ها (Features) برای آموزش بهتر مدل
    # نکته: astype("float32") برای بهینه‌سازی حافظه در زمان آموزش

    data_array = df.values.astype("float32")

    # مرحله ۲: پیدا کردن موقعیت عددی ستون هدف
    # دلیل: برای اینکه بدانیم از کدام ستون در آرایه دیتا باید مقدار y را استخراج کنیم
    target_idx = df.columns.get_loc(target_col)

    # مرحله ۳: ایجاد لیست‌های خالی برای ذخیره دنباله‌ها
    X, y = [], []

    # مرحله ۴: ساخت پنجره لغزان (Sliding Window) روی کل داده‌ها
    # منطق: در هر تکرار، یک بلاک از تمام ویژگی‌ها را برای ورودی برمی‌داریم
    for i in range(len(data_array) - look_back):
        # استخراج ورودی: تمام ستون‌ها از ردیف i تا i+look_back
        # X شامل (look_back × تعداد کل ستون‌ها) می‌شود
        X.append(data_array[i: i + look_back, :])

        # استخراج هدف: مقدار ستون هدف در ردیف بلافاصله بعد از پنجره
        y.append(data_array[i + look_back, target_idx])

    # مرحله ۵: تبدیل لیست‌ها به آرایه‌های numpy برای سازگاری با Keras
    X = np.array(X)
    y = np.array(y)

    # نکته مهم: در اینجا X به صورت خودکار شکل (samples, look_back, num_features) دارد.
    # به دلیل اینکه ورودی ما در مرحله ۴ تمام ستون‌ها بوده، دیگر نیازی به reshape دستی نیست.

    return X, y


def plot_final_results(y_true, y_pred, target_col, history):
    """
    ایجاد داشبورد جامع بصری‌سازی نتایج مدل پیش‌بینی ترافیک شبکه.

    این تابع ۶ نمودار مختلف ایجاد می‌کند تا عملکرد مدل را از جنبه‌های
    مختلف تحلیل کند. داشبورد ایجاد شده شامل ارزیابی پیش‌بینی‌ها، خطاها،
    و روند آموزش مدل می‌باشد.

    ساختار داشبورد:
    ---------------
    ردیف اول:
    ۱. نمودار پراکندگی پیش‌بینی‌ها در مقابل مقادیر واقعی
    ۲. مقایسه ۱۰۰ نمونه اول واقعی و پیش‌بینی شده
    ۳. توزیع خطاها (رزیدوال)

    ردیف دوم:
    ۴. روند خطای آموزش و اعتبارسنجی در دوره‌های مختلف
    ۵. روند MAE آموزش و اعتبارسنجی در دوره‌های مختلف
    ۶. توزیع خطاهای مطلق به صورت Boxplot

    پارامترها:
    -----------
    y_true : numpy.ndarray یا list
        مقادیر واقعی هدف (ground truth)
        باید هم‌شکل با y_pred باشد.

    y_pred : numpy.ndarray یا list
        مقادیر پیش‌بینی شده توسط مدل
        باید هم‌شکل با y_true باشد.

    target_col : str
        نام ستون هدف (برای عنوان‌گذاری نمودارها)

    history : tensorflow.keras.callbacks.History
        آبجکت history بازگشتی از تابع fit مدل
        حاوی اطلاعات loss و metrics در هر epoch

    بازگشت:
    -------
    None
        تابع نمودارها را نمایش می‌دهد و در فایل ذخیره می‌کند.

    خروجی‌های جانبی:
    ----------------
    ۱. نمایش ۶ نمودار در یک پنجره
    ۲. ذخیره نمودارها در فایل 'final_results.png' با کیفیت بالا

    منطق طراحی داشبورد:
    -------------------
    - ترکیب نمودارهای تحلیلی و مقایسه‌ای
    - تمرکز همزمان بر دقت پیش‌بینی و روند آموزش
    - ارائه دیدگاه‌های کمی و کیفی
    - استفاده از معیارهای مختلف ارزیابی (MSE, MAE, Residuals)

    نکات:
    -----
    ۱. برای داده‌های با حجم زیاد، نمودار نمونه‌ها به ۱۰۰ نمونه اول محدود می‌شود
    ۲. نمودارها با dpi=300 ذخیره می‌شوند تا کیفیت چاپ مناسب باشد
    ۳. از gridهای نیمه‌شفاف برای خوانایی بهتر استفاده شده است
    ۴. رنگ‌بندی به گونه‌ای انتخاب شده که برای افراد با کوررنگی نیز قابل تشخیص باشد
    """

    # مرحله ۱: ایجاد figure با اندازه مناسب
    # دلیل: figure 15x10 اینچ برای نمایش همزمان ۶ نمودار با خوانایی مناسب
    plt.figure(figsize=(15, 10))

    # نمودار ۱: نمودار پراکندگی پیش‌بینی‌ها در مقابل مقادیر واقعی
    # هدف: بررسی همبستگی کلی و شناسایی systematic bias
    plt.subplot(2, 3, 1)  # موقعیت: ردیف ۱، ستون ۱

    # رسم نقاط پراکندگی
    # دلیل: alpha=0.3 برای نمایش تراکم نقاط در مناطق پرتراکم
    # دلیل: s=10 برای اندازه نقاط متعادل (نه خیلی بزرگ، نه خیلی کوچک)
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)

    # محاسبه محدوده مقادیر برای خط مبنا
    # دلیل: ایجاد خط y=x برای مقایسه (پیش‌بینی ایده‌آل روی این خط قرار می‌گیرد)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    # رسم خط مبنا (خط قرمز نقطه‌چین)
    # دلیل: lw=2 برای ضخامت قابل مشاهده، 'r--' برای رنگ قرمز و خط نقطه‌چین
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    # تنظیمات محورها و عنوان
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted ({target_col})')
    plt.grid(True, alpha=0.3)  # grid نیمه‌شفاف برای خوانایی بهتر

    # نمودار ۲: مقایسه ۱۰۰ نمونه اول واقعی و پیش‌بینی شده
    # هدف: بررسی کیفیت پیش‌بینی در سطح نمونه‌های فردی
    plt.subplot(2, 3, 2)  # موقعیت: ردیف ۱، ستون ۲

    # تعیین اندازه نمونه برای نمایش
    # دلیل: محدود کردن به ۱۰۰ نمونه برای جلوگیری از شلوغی نمودار
    sample_size = min(100, len(y_true))
    indices = range(sample_size)

    # رسم مقادیر واقعی (خط آبی)
    # دلیل: alpha=0.8 برای شفافیت مناسب
    plt.plot(indices, y_true[:sample_size], 'b-', label='Actual', alpha=0.8)

    # رسم مقادیر پیش‌بینی شده (خط قرمز)
    # دلیل: alpha=0.6 برای تفکیک از خط آبی
    plt.plot(indices, y_pred[:sample_size], 'r-', label='Predicted', alpha=0.6)

    # تنظیمات محورها و عنوان
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'First {sample_size} Samples')
    plt.legend()  # نمایش راهنما برای تشخیص خطوط
    plt.grid(True, alpha=0.3)

    # نمودار ۳: هیستوگرام توزیع خطاها (رزیدوال)
    # هدف: تحلیل توزیع خطاها و بررسی نرمال بودن آنها
    plt.subplot(2, 3, 3)  # موقعیت: ردیف ۱، ستون ۳

    # محاسبه رزیدوال‌ها (تفاوت مقادیر واقعی و پیش‌بینی شده)
    # دلیل: رزیدوال‌های نزدیک به صفر نشان‌دهنده پیش‌بینی دقیق هستند
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true length ({len(y_true)}) != y_pred length ({len(y_pred)})")

    residuals = y_true - y_pred

    # رسم هیستوگرام رزیدوال‌ها
    # دلیل: bins=50 برای جزئیات مناسب در توزیع
    # دلیل: edgecolor='black' برای تشخیص مرزبین‌ها
    # دلیل: alpha=0.7 برای شفافیت مناسب
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)

    # تنظیمات محورها و عنوان
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)

    # نمودار ۴: روند خطای آموزش و اعتبارسنجی
    # هدف: مانیتورینگ فرآیند آموزش و تشخیص overfitting/underfitting
    plt.subplot(2, 3, 4)  # موقعیت: ردیف ۲، ستون ۱

    # بررسی وجود کلیدها در history
    if 'loss' not in history.history or 'val_loss' not in history.history:
        print("Warning: Training history incomplete, skipping loss plots")
        return
    # رسم خطای آموزش
    # دلیل: پیگیری کاهش loss در طول دوره‌های آموزش
    plt.plot(history.history['loss'], label='Train Loss')

    # رسم خطای اعتبارسنجی
    # دلیل: بررسی عملکرد مدل بر روی داده‌های دیده نشده
    plt.plot(history.history['val_loss'], label='Val Loss')

    # تنظیمات محورها و عنوان
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History - Loss')
    plt.legend()  # نمایش راهنما
    plt.grid(True, alpha=0.3)

    # نمودار ۵: روند MAE آموزش و اعتبارسنجی
    # هدف: بررسی معیار MAE که تفسیر آن ساده‌تر از MSE است
    plt.subplot(2, 3, 5)  # موقعیت: ردیف ۲، ستون ۲

    # رسم MAE آموزش
    plt.plot(history.history['mae'], label='Train MAE')

    # رسم MAE اعتبارسنجی
    plt.plot(history.history['val_mae'], label='Val MAE')

    # تنظیمات محورها و عنوان
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training History - MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # نمودار ۶: Boxplot توزیع خطاهای مطلق
    # هدف: نمایش آمارهای توصیفی خطاها (میانگین، میانه، outliers)
    plt.subplot(2, 3, 6)  # موقعیت: ردیف ۲، ستون ۳

    # محاسبه خطاهای مطلق
    # دلیل: خطای مطلق همیشه مثبت است و تفسیر آن ساده‌تر است
    errors = np.abs(y_true - y_pred)

    # رسم Boxplot
    # دلیل: نمایش میانه، چارک‌ها، و outliers در یک نگاه
    plt.boxplot(errors)

    # تنظیمات محور و عنوان
    plt.ylabel('Absolute Error')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)

    # مرحله ۲: تنظیم فاصله‌گذاری بین نمودارها
    # دلیل: tight_layout برای جلوگیری از هم‌پوشانی عناصر
    plt.tight_layout()

    # مرحله ۳: ذخیره نمودار در فایل
    # دلیل: dpi=300 برای کیفیت بالا و قابلیت چاپ
    # دلیل: bbox_inches='tight' برای حذف حاشیه‌های اضافی
    plt.savefig('final_results.png', dpi=300, bbox_inches='tight')

    # مرحله ۴: نمایش نمودار
    # دلیل: نمایش نتایج به کاربر
    plt.show()


def train_or_finetune_model(raw_csv_path: str, is_new_data: bool = True):
    """
    خط کامل آموزش، fine-tuning یا بارگذاری مدل پیش‌بینی ترافیک شبکه.

    این تابع اصلی ترین پالپاین سیستم را پیاده‌سازی می‌کند که شامل:
    ۱. تشخیص تغییرات در داده‌ها (با استفاده از مقایسه هش)
    ۲. پیش‌پردازش خودکار داده‌های CSV
    ۳. انتخاب هوشمند ستون هدف
    ۴. ایجاد دنباله‌های زمانی
    ۵. تقسیم داده به train/validation/test
    ۶. تصمیم‌گیری برای آموزش جدید، fine-tuning یا بارگذاری مدل موجود
    ۷. آموزش یا fine-tuning مدل GRU
    ۸. ارزیابی جامع مدل
    ۹. ذخیره و گزارش‌دهی نتایج

    پارامترها:
    -----------
    raw_csv_path : str
        مسیر فایل CSV خام حاوی داده‌های ترافیک شبکه

    is_new_data : bool, optional
        نشان‌دهنده جدید بودن داده‌ها (پیش‌فرض: True)
        اگر True باشد: آموزش جدید یا fine-tuning انجام می‌شود
        اگر False باشد: مدل موجود بارگذاری و ارزیابی می‌شود

    بازگشت:
    -------
    dict
        دیکشنری حاوی تمام اطلاعات و نتایج پالپاین:
        - 'model': مدل آموزش‌دیده یا بارگذاری‌شده
        - 'target': نام ستون هدف انتخاب‌شده
        - 'metrics': معیارهای ارزیابی (MSE, MAE, R²)
        - 'df': دیتافریم پاک‌شده
        - 'is_new_data': وضعیت جدید بودن داده‌ها
        - 'trained': آیا مدل آموزش دیده است یا فقط بارگذاری شده

    منطق تصمیم‌گیری پالپاین:
    -----------------------
    ۱. اگر داده جدید باشد (is_new_data=True):
        - اگر مدل قبلی وجود داشته باشد: fine-tuning روی داده‌های جدید
        - اگر مدل قبلی وجود نداشته باشد: آموزش مدل جدید از صفر

    ۲. اگر داده جدید نباشد (is_new_data=False):
        - بارگذاری مدل موجود
        - ارزیابی مدل روی داده‌های تست
        - بدون آموزش مجدد

    مراحل اصلی پالپاین:
    -------------------
    ۱. پیش‌پردازش داده‌ها
    ۲. انتخاب ستون هدف
    ۳. ایجاد دنباله‌های زمانی
    ۴. تقسیم داده‌ها
    ۵. ساخت/بارگذاری مدل
    ۶. آموزش/ارزیابی
    ۷. گزارش‌دهی نتایج

    نکات:
    -----
    - از EarlyStopping برای جلوگیری از overfitting استفاده می‌شود
    - از ReduceLROnPlateau برای تنظیم خودکار نرخ یادگیری
    - از ModelCheckpoint برای ذخیره بهترین مدل
    - داده‌ها به صورت زمانی تقسیم می‌شوند (shuffle=False)
    - معیارهای ارزیابی جامع شامل MSE, MAE, R² می‌شود
    """
    # ===================================================================================================================
    # مرحله ۱: شروع پالپاین با نمایش پیام آغازین
    # دلیل: اطلاع‌رسانی به کاربر درباره شروع فرآیند
    print("TRAINING PIPELINE")
    mse = mae = r2 = None
    # نمایش وضعیت داده‌ها
    # دلیل: شفاف‌سازی درباره تصمیم‌گیری‌های بعدی پالپاین
    if is_new_data:
        print("New dataset detected - Starting training/fine-tuning")
    else:
        print("Dataset unchanged - Loading existing model")
    # ==================================================================================================================
    # مرحله ۲: پیش‌پردازش داده‌های CSV
    # دلیل: پاک‌سازی، مدیریت مقادیر گم‌شده و آماده‌سازی داده‌ها
    # نکته: verbose=True برای نمایش جزئیات فرآیند پیش‌پردازش
    df_clean = auto_preprocess_csv(raw_csv_path, verbose=True)
    # ==================================================================================================================
    # مرحله ۳: انتخاب هوشمند ستون هدف
    # دلیل: تحلیل خودکار ستون‌ها و انتخاب بهترین ستون برای پیش‌بینی
    # استفاده از FeatureAnalyzer به جای تابع قدیمی
    print("ANALYZING FEATURES FOR TARGET SELECTION")

    # ایجاد نمونه‌ای از FeatureAnalyzer
    feature_analyzer = FeatureAnalyzer(verbose=True)

    # تحلیل و انتخاب ستون هدف
    target_column = feature_analyzer.analyze_dataframe(df_clean)

    # دریافت رتبه‌بندی ستون‌ها
    column_ranking = feature_analyzer.get_column_ranking()
    if not column_ranking.empty:
        print("\nTop 10 columns by score:")
        print(column_ranking.head(10).to_string(index=False))
    # ==================================================================================================================
    # مرحله ۴: ایجاد دنباله‌های زمانی برای مدل GRU
    # دلیل: تبدیل داده‌های سری زمانی به فرمت مناسب برای شبکه‌های عصبی بازگشتی

    print(f"\nCreating sequences for '{target_column}' with look_back={LOOK_BACK}...")
    X, y = create_simple_sequences(df_clean, target_column, LOOK_BACK)
    if len(X) == 0:
        raise ValueError(
            f"No sequences created! look_back={LOOK_BACK} is too large for dataset with {len(df_clean)} rows")

    # نمایش ابعاد داده‌های ایجاد شده
    # دلیل: بررسی صحت فرآیند ایجاد دنباله‌ها
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    print(f"   Total sequences: {len(X):,}")
    # ==================================================================================================================
    # مرحله ۵: تقسیم داده‌ها به train/validation/test
    # دلیل: ارزیابی عادلانه مدل و جلوگیری از data leakage
    # نکته ۱: shuffle=False برای حفظ ترتیب زمانی داده‌ها
    # نکته ۲: random_state=42 برای تکرارپذیری نتایج
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False, random_state=42
    )

    # تقسیم train_val به train و validation
    # دلیل: نیاز به مجموعه validation برای تنظیم hyperparameters و early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SIZE, shuffle=False, random_state=42
    )

    # نمایش آمار تقسیم داده‌ها
    # دلیل: شفاف‌سازی درباره حجم داده‌های آموزشی و ارزیابی
    print(f"\nData Split:")
    print(f"   Training:   {len(X_train):,} samples ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"   Validation: {len(X_val):,} samples ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"   Test:       {len(X_test):,} samples ({len(X_test) / len(X) * 100:.1f}%)")

    # تعریف شکل ورودی مدل
    # دلیل: مدل GRU به شکل (timesteps, features) نیاز دارد
    input_shape = (X.shape[1], X.shape[2])

    # مقداردهی اولیه متغیرها
    # دلیل: جلوگیری از ارجاع به متغیرهای تعریف‌نشده
    model = None
    history_obj = None
    trained = False
    # ==================================================================================================================
    # مرحله ۶: تصمیم‌گیری درباره بارگذاری یا آموزش مدل
    # منطق: اگر مدل موجود باشد و داده جدید نباشد، مدل را بارگذاری کن
    if os.path.exists(MODEL_PATH) and not is_new_data:
        print(f"\nLoading existing model from {MODEL_PATH}")
        try:
            # تلاش برای بارگذاری مدل موجود
            # دلیل: compile=True برای بارگذاری با همان تنظیمات کامپایل
            model = load_model(MODEL_PATH, compile=True)
            print("Model loaded successfully!")
            # ==================================================================================================================
            # مرحله ۷: ارزیابی مدل بارگذاری‌شده
            # دلیل: بررسی عملکرد مدل روی داده‌های تست
            print(f"\nMaking predictions with existing model...")
            y_pred = model.predict(X_test).reshape(-1)

            # محاسبه معیارهای ارزیابی
            # دلیل: اندازه‌گیری عملکرد مدل با معیارهای استاندارد
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # ایجاد یک آبجکت ساختگی برای history
            # دلیل: سازگاری با تابع plot_final_results در صورت نیاز
            history_obj = type('obj', (object,), {'history': {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}})()
            trained = False

        except Exception as e:
            # خطا در بارگذاری مدل
            # دلیل: فایل مدل ممکن است خراب باشد یا نسخه‌های TensorFlow متفاوت باشد
            print(f"Error loading model: {e}")
            print("Building new model...")
            is_new_data = True  # تغییر وضعیت به "داده جدید" برای ساخت مدل جدید
    # ==================================================================================================================
    # مرحله ۸: ساخت و آموزش مدل جدید (در صورت نیاز)
    if is_new_data or model is None:
        trained = True  # پرچم نشان‌دهنده آموزش مدل
        print(f"\nBuilding model with input shape: {input_shape}")

        # بررسی وجود مدل قبلی برای fine-tuning
        # منطق: اگر مدل قبلی وجود داشته باشد، آن را برای fine-tuning بارگذاری کن
        if os.path.exists(MODEL_PATH):
            print("Fine-tuning existing model on new data...")
            try:
                # بارگذاری مدل موجود برای fine-tuning
                # دلیل: استفاده از دانش قبلی مدل برای یادگیری سریع‌تر روی داده‌های جدید
                model = load_model(MODEL_PATH, compile=True)
                print("Loaded existing model for fine-tuning")

                # تعیین تعداد epoch برای fine-tuning
                # دلیل: fine-tuning معمولاً به epoch کمتری نیاز دارد
                epochs = EPOCHS_FINE_TUNE
                print(f"Fine-tuning for {epochs} epochs...")

            except Exception as e:
                # خطا در بارگذاری مدل برای fine-tuning
                # دلیل: ساخت مدل کاملاً جدید در صورت عدم موفقیت
                print(f"Error loading model for fine-tuning: {e}")
                print("Building completely new model...")
                model = build_final_model(input_shape)
                epochs = EPOCHS_NEW_MODEL
        else:
            # ساخت مدل کاملاً جدید
            # دلیل: عدم وجود مدل قبلی برای بارگذاری
            print("Building new model (no existing model found)...")
            model = build_final_model(input_shape)
            epochs = EPOCHS_NEW_MODEL

        # نمایش خلاصه مدل
        # دلیل: بررسی ساختار مدل و تعداد پارامترها
        model.summary()
        # ==================================================================================================================
        # مرحله ۹: آموزش مدل
        print(f"\nTraining for {epochs} epochs...")

        # تعریف callbacks برای بهبود فرآیند آموزش
        # دلیل: بهینه‌سازی خودکار فرآیند آموزش و جلوگیری از overfitting
        callbacks = [
            # EarlyStopping: توقف آموزش در صورت عدم بهبود validation loss
            # دلیل: جلوگیری از overfitting و صرفه‌جویی در زمان
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',  # نظارت بر validation loss
                patience=15,  # صبر ۱۵ epoch قبل از توقف
                restore_best_weights=True,  # بازگردانی بهترین وزن‌ها
                verbose=1  # نمایش پیام
            ),

            # ReduceLROnPlateau: کاهش نرخ یادگیری در صورت توقف بهبود
            # دلیل: کمک به همگرایی بهتر در مراحل پایانی آموزش
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',  # نظارت بر validation loss
                factor=0.5,  # ضریب کاهش نرخ یادگیری
                patience=5,  # صبر ۵ epoch قبل از کاهش
                min_lr=1e-6,  # حداقل نرخ یادگیری
                verbose=1  # نمایش پیام
            ),

            # ModelCheckpoint: ذخیره بهترین مدل در طول آموزش
            # دلیل: حفظ بهترین حالت مدل حتی در صورت overfitting بعدی
            tf.keras.callbacks.ModelCheckpoint(
                MODEL_PATH,  # مسیر ذخیره‌سازی
                save_best_only=True,  # فقط ذخیره بهترین مدل
                monitor='val_loss',  # نظارت بر validation loss
                mode='min',  # کمینه‌سازی loss
                verbose=1  # نمایش پیام
            )
        ]

        # شروع فرآیند آموزش
        # دلیل: آموزش مدل روی داده‌های train با نظارت بر validation
        history_obj = model.fit(
            X_train, y_train,  # داده‌های آموزشی
            validation_data=(X_val, y_val),  # داده‌های validation
            epochs=epochs,  # تعداد دوره‌های آموزش
            batch_size=BATCH_SIZE,  # اندازه batch
            callbacks=callbacks,  # callbacks تعریف‌شده
            verbose=1  # نمایش پیشرفت آموزش
        )

        # ذخیره مدل نهایی
        # دلیل: استفاده در اجراهای بعدی بدون نیاز به آموزش مجدد
        model.save(MODEL_PATH)
        print(f"\nModel saved to: {MODEL_PATH}")
        # ==================================================================================================================
        # مرحله ۱۰: ارزیابی مدل آموزش‌دیده
        # دلیل: اندازه‌گیری عملکرد نهایی مدل روی داده‌های تست
        print(f"\nMaking predictions...")
        y_pred = model.predict(X_test).reshape(-1)

        # محاسبه معیارهای ارزیابی
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    # ==================================================================================================================
    # مرحله ۱۱: گزارش‌دهی جامع نتایج
    print("FINAL EVALUATION")
    print(f"Target Column: {target_column}")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")

    # تفسیر کیفیت مدل بر اساس R²
    # دلیل: ارائه بازخورد کیفی به کاربر درباره عملکرد مدل
    if r2 > 0.8:
        print("EXCELLENT! Model explains over 80% of variance")
    elif r2 > 0.6:
        print("GOOD! Model explains over 60% of variance")
    elif r2 > 0.4:
        print("FAIR! Model explains over 40% of variance")
    elif r2 > 0:
        print("POOR! Model has some predictive power")
    else:
        print("VERY POOR! Model worse than simple average")
    # ==================================================================================================================
    # مرحله ۱۲: مقایسه با baseline ساده
    # دلیل: نمایش ارزش افزوده مدل پیچیده نسبت به روش‌های ساده
    baseline_mae = np.mean(np.abs(y_test - np.mean(y_test)))
    improvement = (baseline_mae - mae) / baseline_mae * 100

    print(f"\nBaseline (mean predictor) MAE: {baseline_mae:.4f}")
    print(f"Model MAE: {mae:.4f}")
    print(f"Improvement over baseline: {improvement:.1f}%")
    # ==================================================================================================================
    # مرحله ۱۳: تحلیل خطاها (رزیدوال)
    # دلیل: درک عمیق‌تر از نحوه خطاکردن مدل
    residuals = y_test - y_pred
    print(f"\nResiduals Analysis:")
    print(f"  Mean of residuals: {np.mean(residuals):.6f}")
    print(f"  Std of residuals: {np.std(residuals):.6f}")
    print(f"  Residuals range: [{residuals.min():.4f}, {residuals.max():.4f}]")

    # محاسبه دقت پیش‌بینی در یک tolerance مشخص
    # دلیل: ارائه معیار عملیاتی برای کاربر
    tolerance = 0.1 * np.std(y_test)
    accurate_predictions = np.mean(np.abs(residuals) < tolerance) * 100
    print(f"\nPredictions within {tolerance:.4f} tolerance: {accurate_predictions:.1f}%")
    # ==================================================================================================================
    # مرحله ۱۴: رسم نمودارهای تحلیلی (اگر آموزش انجام شده باشد)
    # دلیل: ارائه دید بصری از عملکرد مدل
    if trained and history_obj and hasattr(history_obj, 'history') and len(history_obj.history['loss']) > 0:
        plot_final_results(y_test, y_pred, target_column, history_obj)
    elif trained:  # اگر مدل جدید آموزش داده شده، ولی history_obj خالی است
        print("Warning: No training history available for plotting.")
    """" دنباله‌های زمانی با look_back=128 می‌سازد و داده را به نسبت 80-10-10 تقسیم می‌کند. اگر داده تغییر نکرده باشد، مدل موجود را بارگذاری می‌کند. اگر داده جدید باشد، بسته به وجود مدل قبلی، fine-tuning یا آموزش از صفر انجام می‌دهد. در طول آموزش از EarlyStopping، ReduceLROnPlateau و ModelCheckpoint استفاده می‌کند. در نهایت با معیارهای MSE، MAE و R² ارزیابی کرده و با baseline مقایسه می‌کند."

"""
    # ==================================================================================================================
    # مرحله ۱۵: بازگشت نتایج به صورت ساختاریافته
    # دلیل: امکان استفاده از نتایج در مراحل بعدی یا export کردن
    return {
        'model': model,  # مدل نهایی
        'target': target_column,  # ستون هدف انتخاب‌شده
        'metrics': {'mse': mse, 'mae': mae, 'r2': r2},  # معیارهای کمی
        'df': df_clean,  # داده‌های پاک‌شده
        'is_new_data': is_new_data,  # وضعیت جدید بودن داده‌ها
        'trained': trained,  # وضعیت آموزش مدل
        'ranking': column_ranking  # سیستم رتبه‌بندی ستون‌ها

    }


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    raw_csv_path = "network_traffic_dataset.csv"

    print("Starting Intelligent Training Pipeline...")

    is_new_data = is_new_dataset_sha256(raw_csv_path)

    if is_new_data:
        print("New dataset detected!")
    else:
        print("Dataset unchanged (using existing model)")

    try:
        result = train_or_finetune_model(raw_csv_path, is_new_data)

        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Final Results:")
        print(f"   Target Column: {result['target']}")
        print(f"   R² Score: {result['metrics']['r2']:.4f}")
        print(f"   MAE: {result['metrics']['mae']:.4f}")
        print(f"   Model saved: {MODEL_PATH}")

        if result['trained']:
            print(f"   Training: {'Fine-tuned' if os.path.exists(MODEL_PATH) else 'Trained from scratch'}")
        else:
            print(f"   Training: Used existing model (no training needed)")

        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
