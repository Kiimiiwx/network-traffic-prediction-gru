import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import re
from feature_analyzer import FeatureAnalyzer


class DataPreprocessor:
    """
    کلاس اصلی پیش‌پردازش داده‌های ترافیک شبکه

    این کلاس مسئولیت‌های زیر را بر عهده دارد:
    ۱. پالایش ستون‌های بی‌ارزش  
    ۲. مدیریت داده‌های گمشده
    ۳. حذف outlierها
    ۴. تبدیل داده‌های کیفی به کمی
    ۵. نرمال‌سازی داده‌ها
    """

    def __init__(self, verbose: bool = True):
        """
        مقداردهی اولیه پیش‌پردازشگر

        پارامترها:
        -----------
        verbose : bool
            نمایش جزئیات پردازش (پیش‌فرض: True)
        """
        self.verbose = verbose
        self.feature_analyzer = FeatureAnalyzer(verbose=verbose)
        self.label_encoders = {}
        self.scaler = MinMaxScaler()

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        پیش‌پردازش کامل DataFrame

        این تابع تمام مراحل پیش‌پردازش را به ترتیب اجرا می‌کند.

        پارامترها:
        -----------
        df : pd.DataFrame
            دیتافریم خام ورودی

        بازگشت:
        -------
        pd.DataFrame
            دیتافریم پاک‌شده و نرمال‌شده
        """
        if self.verbose:
            print("STARTING DATA PREPROCESSING")
            print(f"Initial shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"Memory usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB")

        # مرحله ۱: حذف ستون‌های بی‌ارزش
        df = self._remove_useless_columns(df)

        # مرحله ۲: مدیریت داده‌های گمشده
        df = self._handle_missing_values(df)

        # مرحله ۳: حذف outlierها
        df = self._remove_outliers(df)

        # مرحله ۴: تبدیل داده‌های کیفی به کمی
        df = self._encode_categorical_columns(df)

        # مرحله ۵: نرمال‌سازی داده‌ها
        df = self._normalize_numerical_columns(df)

        if self.verbose:
            print("PREPROCESSING COMPLETED")
            print(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"All data types are now numerical")
            print(f"Sample of processed data (first 3 rows):")
            print(df.head(3))

        return df

    def _remove_useless_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        حذف ستون‌های بی‌ارزش از DataFrame

        پارامترها:
        -----------
        df : pd.DataFrame
            دیتافریم ورودی

        بازگشت:
        -------
        pd.DataFrame
            دیتافریم بدون ستون‌های بی‌ارزش
        """
        cols_to_drop = []

        if self.verbose:
            print(f"\n Analyzing columns for uselessness...")

        for col in df.columns:
            if self._is_column_useless(df[col]):
                cols_to_drop.append(col)

        if cols_to_drop:
            df_cleaned = df.drop(columns=cols_to_drop)
            if self.verbose:
                print(f"   Dropped {len(cols_to_drop)} useless columns:")
                for col in cols_to_drop:
                    print(f"     - {col}")
                print(f"   New shape: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
            return df_cleaned

        if self.verbose:
            print(f"   No useless columns found to drop")

        return df

    def _is_column_useless(self, series: pd.Series) -> bool:
        """
        بررسی بی‌ارزش بودن یک ستون

        پارامترها:
        -----------
        series : pd.Series
            ستون مورد بررسی

        بازگشت:
        -------
        bool
            True اگر ستون بی‌ارزش باشد
        """
        col_name = str(series.name).lower() if hasattr(series, 'name') else ''

        # استثنا ۱: ستون‌های زمانی (بعداً پردازش می‌شوند)
        time_keywords = FeatureAnalyzer.TIME_KEYWORDS
        if any(keyword in col_name for keyword in time_keywords):
            return False

        # استثنا ۲: ستون‌های مرتبط با ترافیک شبکه
        traffic_keywords = [kw for kw, _ in FeatureAnalyzer.TRAFFIC_KEYWORDS]
        if series.dtype in ['float64', 'int64']:
            if any(keyword in col_name for keyword in traffic_keywords):
                return False

        # استثنا ۳: ستون‌های وضعیت شبکه
        network_status_keywords = FeatureAnalyzer.NETWORK_STATUS_KEYWORDS
        if any(keyword in col_name for keyword in network_status_keywords):
            return False

        # استثنا ۴: پورت‌های شبکه
        if series.dtype in ['float64', 'int64'] and 2 <= series.nunique() <= 100:
            port_keywords = ['port', 'srcport', 'dstport', 'sport', 'dport']
            if any(keyword in col_name for keyword in port_keywords):
                return False

        # فیلتر ۱: تنوع صفر یا ناچیز
        if series.nunique() <= 1:
            if self.verbose:
                print(f"     - {series.name}: Only {series.nunique()} unique value(s)")
            return True

        # فیلتر ۲: داده‌های گمشده بیش از حد
        missing_ratio = series.isna().mean()
        if missing_ratio > 0.7:
            if self.verbose:
                print(f"     - {series.name}: {missing_ratio:.1%} missing data")
            return True

        # فیلتر ۳: تغییرات ناچیز در داده‌های عددی
        if series.dtype in ['float64', 'int64']:
            value_range = series.max() - series.min()
            if value_range > 0:
                cv = series.std() / value_range
                if cv < 0.0001:
                    if self.verbose:
                        print(f"     - {series.name}: Negligible variation (CV={cv:.6f})")
                    return True

        # فیلتر ۴: شناسه‌های یکتا
        unique_ratio = series.nunique() / len(series)
        if unique_ratio > 0.95:
            important_network_features = ['port', 'protocol', 'service', 'type', 'class']
            if not any(feature in col_name for feature in important_network_features):
                if self.verbose:
                    print(f"     - {series.name}: {unique_ratio:.1%} unique values (likely an ID)")
                return True

        # فیلتر ۵: متن‌های بسیار طولانی
        if series.dtype == "object":
            avg_len = series.astype(str).map(len).mean()
            if avg_len > 200:
                if self.verbose:
                    print(f"     - {series.name}: Long texts (average {avg_len:.0f} characters)")
                return True

        # فیلتر ۶: الگوهای خاص در متن
        if series.dtype == "object" and len(series) > 0:
            if self._contains_structured_patterns(series):
                if self.verbose:
                    print(f"     - {series.name}: Contains structured patterns (URL, IP, JSON, etc.)")
                return True

        return False

    def _contains_structured_patterns(self, series: pd.Series) -> bool:
        """
        بررسی وجود الگوهای ساختاریافته در داده‌های متنی

        پارامترها:
        -----------
        series : pd.Series
            ستون متنی مورد بررسی

        بازگشت:
        -------
        bool
            True اگر الگوهای ساختاریافته پیدا شود
        """
        sample_size = min(10, len(series))
        samples = series.dropna().sample(n=sample_size, random_state=42).astype(str).tolist()

        patterns = [
            (r"https?://", "URL"),
            (r"www\.", "website"),
            (r"\.(com|org|net|ir|edu|gov)", "domain"),
            (r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "IP address"),
            (r"[0-9A-Fa-f]{2}[:-]", "MAC address"),
            (r"\{.*\}", "JSON"),
            (r"\[.*\]", "array"),
            (r"<.*>", "HTML tag"),
            (r"\d{4}-\d{2}-\d{2}", "date"),
            (r"\d{2}:\d{2}:\d{2}", "time")
        ]

        for sample in samples:
            for pattern, pattern_name in patterns:
                if re.search(pattern, sample):
                    return True

        return False

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        مدیریت داده‌های گمشده

        پارامترها:
        -----------
        df : pd.DataFrame
            دیتافریم ورودی

        بازگشت:
        -------
        pd.DataFrame
            دیتافریم بدون داده‌های گمشده
        """
        if self.verbose:
            missing_before = df.isna().sum().sum()
            if missing_before > 0:
                print(f"\nHandling missing values...")
                print(f"   Total missing values: {missing_before}")

                # نمایش ستون‌های با داده‌های گمشده
                missing_cols = df.isna().sum()
                missing_cols = missing_cols[missing_cols > 0]
                for col, count in missing_cols.items():
                    percentage = (count / len(df)) * 100
                    print(f"     - {col}: {count} ({percentage:.2f}%)")
                    if percentage > 10:
                        print(f"       Warning: High missing data percentage")

        df_filled = df.copy()

        for col in df_filled.columns:
            if df_filled[col].isna().any():
                if df_filled[col].dtype in ['float64', 'int64']:
                    # برای مقادیر عددی: میانه
                    df_filled[col].fillna(df_filled[col].median(), inplace=True)
                elif df_filled[col].dtype == 'object':
                    # برای مقادیر متنی: مود
                    if not df_filled[col].mode().empty:
                        df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
                    else:
                        df_filled[col].fillna('Unknown', inplace=True)

        if self.verbose and missing_before > 0:
            missing_after = df_filled.isna().sum().sum()
            if missing_after == 0:
                print(f"   All missing values handled")
            else:
                print(f"   Remaining missing values: {missing_after}")

        return df_filled

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        حذف outlierها با روش IQR

        پارامترها:
        -----------
        df : pd.DataFrame
            دیتافریم ورودی

        بازگشت:
        -------
        pd.DataFrame
            دیتافریم بدون outlierهای شدید
        """
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if not numeric_cols:
            return df

        if self.verbose:
            print(f"\nRemoving outliers from {len(numeric_cols)} numeric columns...")

        df_before = len(df)
        df_clean = df.copy()

        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                outliers_in_col = (~mask).sum()

                if outliers_in_col > 0 and self.verbose:
                    print(f"   {col}: Removed {outliers_in_col} outliers")

                df_clean = df_clean[mask].copy()

        outliers_removed = df_before - len(df_clean)

        if self.verbose and outliers_removed > 0:
            print(f"   Removed {outliers_removed} rows containing outliers")
            print(f"   Data reduction: {(outliers_removed / df_before * 100):.1f}%")
            print(f"   Final shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
        elif self.verbose:
            print(f"   No outliers removed")

        return df_clean

    def _encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        تبدیل ستون‌های کیفی به کمی با Label Encoding

        پارامترها:
        -----------
        df : pd.DataFrame
            دیتافریم ورودی

        بازگشت:
        -------
        pd.DataFrame
            دیتافریم با ستون‌های عددی
        """
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()

        if not object_cols:
            return df

        if self.verbose:
            print(f"\nEncoding {len(object_cols)} categorical columns...")

        df_encoded = df.copy()
        self.label_encoders = {}

        # جدا کردن ستون‌های زمانی
        time_keywords = FeatureAnalyzer.TIME_KEYWORDS
        timestamp_cols = [col for col in object_cols
                          if any(keyword in str(col).lower()
                                 for keyword in time_keywords)]
        other_object_cols = [col for col in object_cols if col not in timestamp_cols]

        # پردازش ستون‌های زمانی
        if timestamp_cols:
            df_encoded = self.feature_analyzer.extract_time_features(df_encoded)

        # پردازش سایر ستون‌های متنی
        if other_object_cols:
            for col in other_object_cols:
                unique_values = df_encoded[col].nunique()

                if unique_values > 50 and self.verbose:
                    print(f"   {col}: {unique_values} unique values - consider one-hot encoding")

                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le

                if self.verbose:
                    print(f"   {col}: {unique_values} unique values → encoded")

        return df_encoded

    def _normalize_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        نرمال‌سازی ستون‌های عددی با MinMaxScaler

        پارامترها:
        -----------
        df : pd.DataFrame
            دیتافریم ورودی

        بازگشت:
        -------
        pd.DataFrame
            دیتافریم با داده‌های نرمال‌شده [0, 1]
        """
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if not numeric_cols:
            return df

        if self.verbose:
            print(f"\nNormalizing {len(numeric_cols)} numerical columns to [0, 1] range...")

        df_normalized = df.copy()

        # استفاده از MinMaxScaler
        self.scaler = MinMaxScaler()
        df_normalized[numeric_cols] = self.scaler.fit_transform(df_normalized[numeric_cols])

        if self.verbose:
            print(f"   All numerical columns normalized")

        return df_normalized


def auto_preprocess_csv(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    خط‌لوله کامل پیش‌پردازش داده‌های CSV برای یادگیری ماشین.

    این تابع داده‌های خام شبکه را به فرمتی تبدیل می‌کند که مدل GRU بتواند روی آن آموزش ببیند.
    هدف اصلی: حذف نویز، کاهش ابعاد و استانداردسازی داده‌ها برای بهبود دقت مدل.

    ورودی:
    -----------
    file_path : str
        مسیر فایل CSV حاوی داده‌های خام ترافیک شبکه
    verbose : bool, optional
        اگر True باشد، جزئیات مراحل پردازش نمایش داده می‌شود (پیش‌فرض: True)
        مفید برای دیباگ و درک فرآیند

    خروجی:
    -------
    pd.DataFrame
        دیتافریم پالایش‌شده، نرمال‌شده و آماده برای ایجاد توالی‌های زمانی
        تمام مقادیر در بازه [0, 1] و بدون داده‌های گمشده
    """

    # مرحله ۱: راه‌اندازی و بارگذاری اولیه
    if verbose:
        print("DATA PREPROCESSING PIPELINE")
        print(f"Input file: {file_path}")

    # بارگذاری دیتاست با مدیریت خطاهای احتمالی
    try:
        df = pd.read_csv(file_path)
        if verbose:
            print(f"Dataset loaded successfully")
            print(f"Initial shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"Memory usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # مرحله ۲: ایجاد پیش‌پردازشگر و اجرای پردازش
    preprocessor = DataPreprocessor(verbose=verbose)
    df_clean = preprocessor.preprocess_dataframe(df)

    # مرحله ۳: خلاصه نتایج نهایی
    if verbose:
        print("FINAL DATASET SUMMARY")
        print(f"   Shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
        print(f"   Total features: {len(df_clean.columns)}")
        print(f"   Data types: All numerical (ready for GRU)")

        print(f"\nStatistical summary:")
        print(df_clean.describe())

        print(f"\nReady for sequence creation and model training")

    return df_clean
