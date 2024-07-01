from flask import Flask, request, jsonify, redirect, url_for
import os
import subprocess
import pandas as pd
import uuid
from inference import Inference  # Make sure to import the Inference class from your module
from database import Database  # Make sure to import the Database class from your module
from PIL import Image
from flask_cors import CORS
import pytz

db = Database()
app = Flask(__name__)
CORS(app)

# Set the folders
root_dir='static'
UPLOAD_FOLDER = os.path.join(root_dir,'input_video')
temp_folder = "temp/"
annotated_folder = "annotated_video"
images_folder = "Cow-Images"

if not os.path.exists(os.path.join(root_dir,temp_folder)):
    os.makedirs(os.path.join(root_dir,temp_folder))

if not os.path.exists(os.path.join(root_dir,annotated_folder)):
    os.makedirs(os.path.join(root_dir,annotated_folder))

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def convert_video(input_name, output_name):
    ffmpeg_cmd = f'ffmpeg -i "{input_name}" -brand mp42 "{output_name}" 2>/dev/null'
    try:
        print("Converting Video")
        subprocess.run(ffmpeg_cmd, shell=True, check=True)
        print(f"Conversion successful. Output file: {output_name}")
        os.remove(input_name)
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

@app.route('/')
def index():
    return jsonify(message='Welcome to the API')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files or request.files['file'].filename=='':
            return jsonify(message="No 'file' provided or selected")

        file = request.files['file']
        filename = str(uuid.uuid4()) +'___' + str(file.filename)
        raw_filename = os.path.join(root_dir,temp_folder, filename)
        file.save(raw_filename)

        if os.path.splitext(filename)[-1] != ".mp4":
            outputfilename = os.path.splitext(filename)[0] + ".mp4"
        else:
            outputfilename = filename

        convert_video(raw_filename, os.path.join(UPLOAD_FOLDER, outputfilename))
        db.insert_cow_Video_Infomation_data(input_video = filename.split('.mp4')[0])

        return jsonify(message='File uploaded successfully', filename=os.path.join(UPLOAD_FOLDER, outputfilename))

    except Exception as e:
        return jsonify(message=f'Error: {str(e)}')

@app.route('/start_inference', methods=['POST'])
def start_inference():
    
    if 'video_name' not in request.form or request.form['video_name']=='':
        return jsonify(message='Please Provide The video_name to test.')

    filename = request.form['video_name'] #"static/input_video/videoname.mp4"

    try:
        annot_file = os.path.join(root_dir,annotated_folder,filename.split('/')[-1])
        raw_annot_file = os.path.join(root_dir,temp_folder,filename.split('/')[-1])

        inf = Inference(filename, raw_annot_file)

        inf.inference()
        convert_video(raw_annot_file , annot_file)

        db.insert_cow_Video_Infomation_data(output_video = os.path.basename(annot_file).split('.mp4')[0])

        raw_activity_video_path = os.path.join(root_dir, 'Activity-Videos', os.path.basename(filename).split('.mp4')[0], 'Raw-Videos')
        encoded_activity_video_path = os.path.join(root_dir, 'Activity-Videos', os.path.basename(filename).split('.mp4')[0])

        for activity_video_name in os.listdir(raw_activity_video_path):
            convert_video(os.path.join(raw_activity_video_path, activity_video_name),
                           os.path.join(encoded_activity_video_path, activity_video_name))

        return jsonify(message='Inference completed', video_name=annot_file)

    except Exception as e:
        return jsonify(message=f'Error: {str(e)}')

@app.route('/video_analytics', methods=['POST', 'GET'])
def video_analytics():
    try:        
        event_data = fetch_events_data().json
        return jsonify(event_data)
    except Exception as e:
        return jsonify(message=f'Error: {str(e)}')

@app.route('/video_information', methods = ['POST','GET'])
def video_information():
    try:
        response = db.get_video_info()
        df = pd.DataFrame(response, columns = ['Video-Name', 'Output-Video', 'uploaded_datetime'])
        if len(response):
            df['Preview-Video'] = df.apply(lambda row: os.path.join(UPLOAD_FOLDER, row['Video-Name']+'.mp4'), axis=1)
            df['Uploaded-Date'] = pd.to_datetime(df['uploaded_datetime']).dt.tz_localize('UTC').dt.tz_convert('US/Central').dt.strftime('%m/%d/%Y')
            df['Uploaded-Time'] = pd.to_datetime(df['uploaded_datetime']).dt.tz_localize('UTC').dt.tz_convert('US/Central').dt.strftime('%I:%M:%S %p')
            df['Video-Name'] = df['Video-Name'].apply(lambda x: f"{x.split('___')[1].strip()}.mp4")
            df = df.drop(columns=['uploaded_datetime'])
            
            df['Inference-Status'] = df['Output-Video'].apply(lambda x: 'To-be-processed' if pd.isnull(x) else 'Processed')
            df = df.drop(columns=['Output-Video'])

        list_of_rows = df.to_dict(orient='records')

        return jsonify(list_of_rows)
    except Exception as e:
        return jsonify(message=f'Error: {str(e)}')



'''
@app.route('/get_cow_thumbnails', methods=['POST', 'GET'])
def get_images_paths():

    if 'video_name' not in request.form or request.form['video_name']=='':
        return jsonify(message='Please Provide The video_name.')
        
    video_name = request.form['video_name'] #videoname.mp4

    try:
        video_name = os.path.basename(video_name).split('.mp4')[0]
        cow_ids, thumbnail_paths = db.get_cow_image_and_thumbnail(video_name)
        
        response_data = {
            'cow_ids': cow_ids,
            'thumbnail_paths': thumbnail_paths
        }
        return jsonify(response_data), 200
    
    except Exception as e:
        return jsonify(message=f'Error: {str(e)}'), 500
'''
@app.route('/get_cow_images', methods=['POST','GET'])
def get_cow_images():
    try:
        cluster_dict = training_image_clusters()
        response = db.get_cow_image_paths()
        df = pd.DataFrame(response, columns = ['Cow-ID', 'Video-Name', 'Date', 'Image-Paths','Cluster'])
        df["Image-Paths"] = df["Image-Paths"].apply(lambda x: x.split(';'))
        df = df.drop(columns=['Cow-ID','Video-Name','Date'])
        #print(cluster_dict)
        for id in cluster_dict:
            if id in df.Cluster.values:
                paths = list(df[df["Cluster"]==id]["Image-Paths"].values)[0]
                cluster_dict[id] = cluster_dict[id] + paths
        if "New" in list(df["Cluster"].values):
            cluster_dict["New"] = list(df[df["Cluster"]=="New"]["Image-Paths"].values)[0]
        #print(cluster_dict)
        df = pd.DataFrame(list(cluster_dict.items()), columns=['Cluster', 'Image-Paths'])
        df['Cluster'] = df['Cluster'].replace('New', 'cluster_999')
        df['Cluster'] = df['Cluster'].str.replace('cluster_', '').astype(int)
        df.sort_values(by='Cluster', inplace=True, ignore_index=True)
        df['Cluster'] = df['Cluster'].replace(999, 'New')
        df.reset_index(drop=True, inplace=True)
        json_data = df.to_dict(orient='records')
        
        return jsonify(json_data)
        
    except Exception as e:
       return jsonify(message=f'Error: {str(e)}'), 500

def training_image_clusters():
    path = os.path.join('static','Dataset')
    image_full_path = {}
    for folder in os.listdir(path):
        for image_path in os.listdir(os.path.join(path, folder)):
            if folder not in image_full_path:
                image_full_path[folder] = [os.path.join(path,folder,image_path)]
            else:
                image_full_path[folder].append(os.path.join(path,folder,image_path))
    #print(image_full_path)
    #cluster_image_df = pd.DataFrame(list(image_full_path.items()), columns=['Cluster', 'Image-Paths'])
    
    return image_full_path


def fetch_events_data():
    df = pd.DataFrame(db.get_events_data(), columns=['Cow-ID', 'Activity-Type', 'Duration', 'Video-Name'])
    cluster_df = pd.DataFrame(db.get_cow_image_paths(), columns=['Cow-ID', 'Video-Name', 'Date', 'Image-Paths', 'Cluster'])
    
    if len(df) and len(cluster_df):
        
        df["Cow-ID"] = df["Cow-ID"].astype(str)
        df['activity_video'] = df.apply(lambda row: os.path.join(root_dir, 'Activity-Videos', row['Video-Name'], f"{row['Cow-ID']}{row['Activity-Type']}.mp4"), axis=1)
        upload_datetime = dict(db.get_video_information_data())
        
        df['uploaded_datetime'] = df['Video-Name'].map(upload_datetime)

        df['Uploaded-Date'] = pd.to_datetime(df['uploaded_datetime']).dt.tz_localize('UTC').dt.tz_convert('US/Central').dt.strftime('%m/%d/%Y')
        df['Uploaded-Time'] = pd.to_datetime(df['uploaded_datetime']).dt.tz_localize('UTC').dt.tz_convert('US/Central').dt.strftime('%I:%M:%S %p')
        df.sort_values(by='uploaded_datetime', inplace=True, ignore_index=True, ascending=False)
        df = df.drop(columns=['uploaded_datetime'])
        cluster_df["Cow-ID"] = cluster_df["Cow-ID"].astype(str)
        df = pd.merge(df, cluster_df[['Cow-ID', 'Video-Name', 'Cluster']], on=['Cow-ID', 'Video-Name'], how='left')
        df['Cluster'].fillna('New', inplace=True)
        df = df.rename(columns={'Cluster': 'Cluster_Id'})
        df = df.drop(columns=['Cow-ID'])
        df['Cluster_Id'] = df['Cluster_Id'].replace('New', 'cluster_999')
        df['Cluster_Id'] = df['Cluster_Id'].str.replace('cluster_', '').astype(int)
        
        df['Cluster_Id'] = df['Cluster_Id'].replace(999, 'New')
        df.reset_index(drop=True, inplace=True)
        df['Video-Name'] = df['Video-Name'].apply(lambda x: f"{x.split('___')[1].strip()}.mp4")
        df = df[df['Duration'] >= 3]
    
    json_data = df.to_dict(orient='records')
   
    return jsonify(json_data)


'''
def fetch_occupancy_data(videoname):
    result = db.get_occupancy_data(videoname.split('.mp4')[0])
    columns = ['frame_number', 'cow_count', 'brush_busy', 'waterbank_busy']
    df = pd.DataFrame(result, columns=columns)
    json_data = df.to_dict(orient='list')
    return jsonify(json_data)
'''

if __name__ == '__main__':
    #print(fetch_events_data())
    app.run(debug=True, host='0.0.0.0', port=5000)
