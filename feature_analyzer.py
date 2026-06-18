"""
feature_analyzer.py

ماژول تحلیل هوشمند ویژگی‌ها برای پیش‌بینی ترافیک شبکه

این ماژول مسئولیت‌های زیر را بر عهده دارد:
۱. تحلیل ستون‌ها برای شناسایی بهترین ستون هدف
۲. مهندسی ویژگی‌های زمانی از ستون‌های timestamp
۳. امتیازدهی و رتبه‌بندی ستون‌ها بر اساس ارتباط با ترافیک شبکه
۴. استخراج ویژگی‌های آماری برای تصمیم‌گیری

"""

import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """
    کلاس اصلی تحلیل ویژگی‌های ترافیک شبکه

    این کلاس روش‌های جامعی برای تحلیل و انتخاب ویژگی‌های مناسب
    برای مدل‌سازی ترافیک شبکه ارائه می‌دهد.
    """

    # کلمات کلیدی مرتبط با ترافیک شبکه به همراه وزن اهمیت
    # وزن‌ها بر اساس ارتباط مستقیم با مفهوم ترافیک شبکه تعیین شده‌اند
    TRAFFIC_KEYWORDS = [
        # --- اولویت اول: ترافیک حجمی و عددی (امتیاز طلایی) ---
        ('visits', 250),  # هدف اصلی شما در اکثر دیتاست‌های وب
        ('website_visits', 250),
        ('traffic_volume', 200),
        ('total_bytes', 180),
        ('bytes_sent', 150),
        ('bytes_received', 150),
        ('packets_count', 150),
        ('bandwidth_usage', 150),
        ('throughput', 150),

        # --- اولویت دوم: نرخ و سرعت (امتیاز بالا) ---
        ('traffic_rate', 100),
        ('bit_rate', 100),
        ('request_rate', 100),
        ('upload_speed', 90),
        ('download_speed', 90),
        ('load', 80),
        ('utilization', 80),

        # --- اولویت سوم: کیفیت سرویس (امتیاز متوسط) ---
        ('latency', 50),
        ('delay', 50),
        ('jitter', 50),
        ('packet_loss', 40),

        # --- اولویت چهارم: کلمات خنثی یا کلی (امتیاز کم) ---
        ('traffic', 30),  # کاهش امتیاز برای جلوگیری از تداخل با کلماتی مثل traffic_type
        ('network', 20),
        ('usage', 20),

        # --- خط قرمز: کلمات ممنوعه (امتیاز منفی شدید برای حذف خودکار) ---
        # این کلمات باعث می‌شوند ستون‌های دسته‌بندی یا متنی هرگز انتخاب نشوند
        ('type', -500),  # مثل network_type یا device_type
        ('mode', -500),  # مثل connection_mode
        ('id', -500),  # مثل user_id یا session_id
        ('status', -400),  # مثل connection_status
        ('category', -400),  # دسته‌بندی‌ها
        ('protocol', -400),  # پروتکل‌ها (TCP/UDP)
        ('index', -400),  # ردیف یا ایندکس
        ('name', -400),  # نام‌ها
        ('address', -400),  # IP یا MAC address
        ('ip', -400),
        ('mac', -400),
        ('is_', -300),  # ستون‌های Boolean مثل is_weekend یا is_holiday
        ('_flag', -300)  # پرچم‌های وضعیتی
    ]

    # کلمات کلیدی برای شناسایی ستون‌های وضعیت شبکه
    NETWORK_STATUS_KEYWORDS = [
        'status', 'state', 'flag', 'protocol',
        'type', 'service', 'port', 'srcport',
        'dstport', 'sport', 'dport'
    ]

    # کلمات کلیدی برای شناسایی ستون‌های زمانی
    TIME_KEYWORDS = ['timestamp', 'time', 'date', 'datetime']

    def __init__(self, verbose: bool = True):
        """
        مقداردهی اولیه تحلیل‌گر ویژگی

        پارامترها:
        -----------
        verbose : bool
            نمایش جزئیات تحلیل (پیش‌فرض: True)
        """
        self.verbose = verbose
        self.best_column = None
        self.column_scores = {}

    def analyze_dataframe(self, df: pd.DataFrame) -> str:
        if self.verbose:
            print("STARTING FEATURE ANALYSIS (INTERACTIVE MODE)")
            print(f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in DataFrame")

        self.analysis_results = []
        for col in numeric_cols:
            # استخراج اطلاعات ستون
            unique_count = len(df[col].unique())
            unique_ratio = unique_count / len(df[col])

            # رفع خطای قبلی: پاس دادن هر دو آرگومان
            variety_score = self._calculate_variety_score(unique_ratio, unique_count, col)

            # محاسبه امتیاز نهایی (ساده شده برای این بخش)
            traffic_score = 0
            for kw, weight in self.TRAFFIC_KEYWORDS:
                if kw in col.lower():
                    traffic_score += weight

            std_val = df[col].std()
            total_score = traffic_score + variety_score + (100 if std_val > 0 else 0)
            self.column_scores[col] = total_score  # ← این خط را اضافه کن

            self.analysis_results.append({
                'Column': col,
                'Score': total_score,
                'Unique_Values': unique_count,
                'Std': std_val
            })

        # تبدیل نتایج به دیتافریم برای رتبه‌بندی
        results_df = pd.DataFrame(self.analysis_results).sort_values(by='Score', ascending=False)

        # منطق جدید: پرسش از کاربر در صورت وجود رقبای نزدیک
        max_score = results_df['Score'].max()
        # کاندیداهایی که امتیازشان به هم نزدیک است (مثلاً در بازه ۱۰ امتیازی بالاترین)
        top_candidates = results_df[results_df['Score'] >= (max_score - 5)].copy()

        if len(top_candidates) > 1:
            print("\nMultiple potential target columns found:")
            for i, row in enumerate(top_candidates.itertuples(), 1):
                suffix = " (Recommended)" if i == 1 else ""
                print(f"   {i}. Column: {row.Column} | Score: {row.Score} | Unique: {row.Unique_Values}{suffix}")

            while True:
                try:
                    choice = input(
                        f"\nSelect target (1-{len(top_candidates)}) or press Enter for '{top_candidates.iloc[0]['Column']}': ").strip()
                    if choice == "":
                        self.best_column = top_candidates.iloc[0]['Column']
                        break
                    idx = int(choice) - 1
                    if 0 <= idx < len(top_candidates):
                        self.best_column = top_candidates.iloc[idx]['Column']
                        break
                    else:
                        print("Invalid number.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            self.best_column = results_df.iloc[0]['Column']

        print(f"\nFINAL CHOICE: '{self.best_column}' selected for training.")
        return self.best_column

    def _calculate_variety_score(self, unique_ratio: float, unique_count: int, col_name: str) -> int:
        """
        امتیازدهی سخت‌گیرانه برای حذف ستون‌های دسته‌بندی شده (Categorical)
        """
        # اگر ستونی کلاً زیر 20 مقدار منحصربه‌فرد داشته باشد، برای رگرسیون ترافیک سَم است!
        if unique_count < 20:
            return -2000

            # بازه طلایی برای ترافیک شبکه (تنوع بین 10% تا 70%)
        if 0.1 <= unique_ratio <= 0.7:
            return 200
        elif unique_ratio > 0.95:
            # استثنا: ستون‌های عددی با تنوع بالا که نام ترافیکی دارند
            traffic_indicators = ['rate', 'volume', 'count', 'bytes', 'packet', 'traffic', 'bandwidth', 'latency']
            if any(indicator in col_name.lower() for indicator in traffic_indicators):
                return 200  # ← این ستون‌ها را تشویق کن، نه جریمه!
            return -1000  # فقط IDهای واقعی را جریمه کن
        else:
            return 50

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        استخراج ویژگی‌های زمانی از ستون‌های timestamp

        پارامترها:
        -----------
        df : pd.DataFrame
            دیتافریم اصلی

        بازگشت:
        -------
        pd.DataFrame
            دیتافریم با ویژگی‌های زمانی اضافه شده
        """
        timestamp_cols = [col for col in df.columns
                          if any(keyword in str(col).lower()
                                 for keyword in self.TIME_KEYWORDS)]

        if not timestamp_cols:
            return df

        df_processed = df.copy()

        for timestamp_col in timestamp_cols:
            try:
                if self.verbose:
                    print(f"\nProcessing timestamp column: {timestamp_col}")

                # تبدیل به datetime
                df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col])

                # استخراج ویژگی‌های زمانی
                df_processed[f'{timestamp_col}_hour'] = df_processed[timestamp_col].dt.hour
                df_processed[f'{timestamp_col}_dayofweek'] = df_processed[timestamp_col].dt.dayofweek
                df_processed[f'{timestamp_col}_is_weekend'] = df_processed[timestamp_col].dt.dayofweek.isin(
                    [5, 6]).astype(int)
                df_processed[f'{timestamp_col}_month'] = df_processed[timestamp_col].dt.month
                df_processed[f'{timestamp_col}_day'] = df_processed[timestamp_col].dt.day

                if self.verbose:
                    print(f"   Created time features:")
                    print(f"     - {timestamp_col}_hour (0-23)")
                    print(f"     - {timestamp_col}_dayofweek (0=Monday, 6=Sunday)")
                    print(f"     - {timestamp_col}_is_weekend (0/1)")
                    print(f"     - {timestamp_col}_month (1-12)")
                    print(f"     - {timestamp_col}_day (1-31)")

                # حذف ستون اصلی timestamp
                df_processed.drop(timestamp_col, axis=1, inplace=True)

            except Exception as e:
                if self.verbose:
                    print(f"   Could not process {timestamp_col}: {e}")
                # حذف ستون در صورت خطا
                df_processed.drop(timestamp_col, axis=1, inplace=True)

        return df_processed

    def get_column_ranking(self) -> pd.DataFrame:
        """
        دریافت رتبه‌بندی ستون‌ها به صورت DataFrame

        بازگشت:
        -------
        pd.DataFrame
            DataFrame حاوی امتیاز و رتبه هر ستون
        """
        if not self.column_scores:
            return pd.DataFrame()

        ranking_df = pd.DataFrame.from_dict(
            self.column_scores,
            orient='index',
            columns=['Score']
        ).reset_index()

        ranking_df.columns = ['Column', 'Score']
        ranking_df = ranking_df.sort_values('Score', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)

        return ranking_df

