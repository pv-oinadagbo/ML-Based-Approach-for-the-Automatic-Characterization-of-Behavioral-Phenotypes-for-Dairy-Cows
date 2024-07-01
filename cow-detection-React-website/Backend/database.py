import sqlite3
import os
from datetime import datetime


class Database:
    def __init__(self):
        self.database_folder = 'Database'
        os.makedirs(self.database_folder, exist_ok = True)
        self.database_name = os.path.join(self.database_folder, "CowsDatabase.db")
        self.cow_events_table = "CowEvents"
        self.cow_occupancy_table = "CowOccupancy"
        self.cow_Images_table = 'CowImages'
        self.cow_Video_Infomation_table = 'VideoInformation'

        if not os.path.exists(self.database_name):
            self.create_database(self.database_name)
            self.create_cow_events_table(
                self.database_name, self.cow_events_table)
            self.create_cow_occupancy_table(
                self.database_name, self.cow_occupancy_table)
            self.create_cow_Images_table(
                self.database_name, self.cow_Images_table)
            self.create_cow_Video_Infomation_table(
                self.database_name, self.cow_Video_Infomation_table)

    def create_database(self, database_name):
        self.conn = sqlite3.connect(database_name)

    def create_cow_events_table(self, database_name, table_name):
        self.conn = sqlite3.connect(database_name)
        self.cursor = self.conn.cursor()
        try:
            self.cursor.execute("CREATE TABLE " + table_name +
                                " (CowID INTEGER, EventType VARCHAR(255), EventTime REAL, VideoName TEXT)")
            self.conn.commit()
        except:
            self.conn.rollback()
        finally:
            self.conn.close()

    def create_cow_occupancy_table(self, database_name, table_name):
        self.conn = sqlite3.connect(database_name)
        self.cursor = self.conn.cursor()
        try:
            self.cursor.execute("CREATE TABLE " + table_name +
                                " (FrameNumber INTEGER, CowCount INTEGER, BrushBusy TEXT, WatertubBusy TEXT, VideoName TEXT)")
            self.conn.commit()
        except:
            self.conn.rollback()
        finally:
            self.conn.close()

    def create_cow_Images_table(self, database_name, table_name):
        self.conn = sqlite3.connect(database_name)
        self.cursor = self.conn.cursor()
        try:
            self.cursor.execute("CREATE TABLE " + table_name +
                                " (CowID INTEGER, VideoName TEXT, ImagePath TEXT, Date TEXT, Cluster TEXT)")
            self.conn.commit()
        except:
            self.conn.rollback()
        finally:
            self.conn.close()

    def create_cow_Video_Infomation_table(self, database_name, table_name):
        self.conn = sqlite3.connect(database_name)
        self.cursor = self.conn.cursor()
        try:
            self.cursor.execute("CREATE TABLE " + table_name +
                                " (InputVideoPath TEXT, OutputVideoPath TEXT, UploadTime TEXT)")
            self.conn.commit()
        except:
            self.conn.rollback()
        finally:
            self.conn.close()

    def insert_cow_events_data(self, cow_id, event_type, event_time, video_name):
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT CowID, EventType, EventTime FROM "+self.cow_events_table+" WHERE CowID = ? AND EventType = ? AND VideoName = ? ", (cow_id, event_type, video_name))
        existing_data = self.cursor.fetchone()
        if existing_data:
            if event_time > 0 and (len(existing_data) < 1 or event_time > existing_data[2]):
                self.cursor.execute("UPDATE " + self.cow_events_table + " SET EventTime = ? WHERE CowID = ? AND EventType = ? AND VideoName = ?", (event_time, cow_id, event_type, video_name))
        else:
            if event_time > 0:
                self.cursor.execute("INSERT INTO " + self.cow_events_table + " (CowID, EventType, EventTime, VideoName) VALUES (?,?,?,?)", (cow_id, event_type, event_time, video_name))
        self.conn.commit()
        self.conn.close()

    def insert_cow_occupancy_data(self, frame_number, cow_count, brush_busy, water_tub_busy, video_name):
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("INSERT INTO " + self.cow_occupancy_table + " (FrameNumber, CowCount, BrushBusy, WatertubBusy, VideoName) VALUES (?,?,?,?,?)",
                            (frame_number, cow_count, brush_busy, water_tub_busy, video_name))
        self.conn.commit()
        self.conn.close()

    def insert_cow_Images_data(self, cow_id, cow_image_path, video_name,cluster_id):
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT ImagePath FROM " + self.cow_Images_table +
                            " WHERE CowID=? AND VideoName=?", (cow_id, video_name))
        existing_paths = self.cursor.fetchone()

        if existing_paths:
            updated_paths = existing_paths[0] + ';' + cow_image_path
            self.cursor.execute("UPDATE " + self.cow_Images_table +
                                " SET ImagePath=?,Cluster=? WHERE CowID=? AND VideoName =?", (updated_paths,cluster_id, cow_id, video_name))
        else:
            self.cursor.execute("INSERT INTO " + self.cow_Images_table +
                                " (CowID, VideoName, ImagePath, Date, Cluster) VALUES (?, ?, ?, ?, ?)", (cow_id, video_name, cow_image_path, datetime.now(), cluster_id))
        self.conn.commit()
        self.conn.close()

    def insert_cow_Video_Infomation_data(self, input_video = None, output_video = None):
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()
        if output_video == None:
            self.cursor.execute("INSERT INTO " + self.cow_Video_Infomation_table +
                                " (InputVideoPath, OutputVideoPath, UploadTime) VALUES (?,?,?)", (input_video, output_video, datetime.now()))
        else:
            self.cursor.execute("UPDATE " + self.cow_Video_Infomation_table +  " SET OutputVideoPath = ? WHERE InputVideoPath = ?",(output_video, output_video))
        self.conn.commit()
        self.conn.close()

    def get_events_data(self):
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT CowID, EventType, EventTime, VideoName FROM " + self.cow_events_table)
        result = self.cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        return result


    def get_occupancy_data(self, video_name):
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT FrameNumber, CowCount, BrushBusy, WatertubBusy FROM " + self.cow_occupancy_table + " WHERE VideoName = ?",(video_name,))
        result = self.cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        return result

    def get_video_names_only(self):
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT InputVideoPath FROM " + self.cow_Video_Infomation_table)
        result = self.cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        results = [i[0] for i in result]
        return results

    def get_cow_image_and_thumbnail(self, video_name):
        #Return Cow IDs and a single thumbnail image associated with them
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT CowID, ImagePath FROM " + self.cow_Images_table + " WHERE VideoName=?",(video_name,))
        result = self.cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        cow_ids = [i[0] for i in result]
        thumbnails = [i[1].split(";")[0] for i in result]
        # print(cow_ids, thumbnails)
        return cow_ids, thumbnails

    def get_cow_image_paths(self):
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT CowID, VideoName, Date, ImagePath, Cluster  FROM " + self.cow_Images_table)
        result = self.cursor.fetchall()
        self.conn.commit()
        self.conn.close()
        return result
        #paths = [i for i in result[][0].split(";")]
        #return paths #list of paths

    def get_video_information_data(self):
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT InputVideoPath, UploadTime FROM " + self.cow_Video_Infomation_table)
        result = self.cursor.fetchall()
        self.conn.commit()
        self.conn.close()

        return result #list of paths

    def get_video_info(self):
        self.conn = sqlite3.connect(self.database_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT InputVideoPath, OutputVideoPath, UploadTime FROM " + self.cow_Video_Infomation_table)
        result = self.cursor.fetchall()
        self.conn.commit()
        self.conn.close()

        return result

if __name__ == "__main__":
    db = Database()
    print(db.get_cow_image_paths())
    