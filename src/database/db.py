import sqlite3
import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

# تنظیم logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path="/app/outputs/production.db"):
        """
        ایجاد ارتباط با دیتابیس SQLite.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.create_table()
        logger.info(f"💾 SQLite Database connected at: {self.db_path}")

    def create_table(self):
        """ایجاد جدول اگر وجود نداشته باشد"""
        query = """
        CREATE TABLE IF NOT EXISTS profile_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_type TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        )
        """
        try:
            self.conn.execute(query)
            self.conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to create SQLite table: {e}")

    def insert_record(self, profile_type: str):
        """
        درج رکورد جدید با تایم‌زون تهران.
        """
        try:
            tehran_tz = ZoneInfo("Asia/Tehran")
            current_time = datetime.now(tehran_tz).strftime('%Y-%m-%d %H:%M:%S')
            query = "INSERT INTO profile_logs (profile_type, timestamp) VALUES (?, ?)"
            self.conn.execute(query, (profile_type, current_time))
            self.conn.commit()
        except Exception as e:
            logger.error(f"❌ Database Insert Error: {e}")

    def get_counts(self, start_time=None, end_time=None):
        """
        خواندن رکوردها و بازگرداندن آن‌ها به صورت لیست دیکشنری شامل تمام فیلدها (id, profile_type, timestamp)
        """
        try:
            cursor = self.conn.cursor()
            # انتخاب صریح ستون‌ها برای اطمینان از وجود timestamp
            query = "SELECT id, profile_type, timestamp FROM profile_logs"
            params = []
            
            conditions = []
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # استخراج نام ستون‌ها از cursor.description
            columns = [description[0] for description in cursor.description]
            
            # تبدیل هر سطر به یک دیکشنری (Key-Value)
            data = []
            for row in rows:
                data.append(dict(zip(columns, row)))
                
            return data
        except Exception as e:
            logger.error(f"❌ Failed to read from database: {e}")
            return []

    def close(self):
        self.conn.close()