import cv2
import os
import face_recognition
import numpy as np
import datetime

def detect_and_crop_faces(image_path):
    # 修改输出文件夹的路径
    output_folder = f"/Users/xigua/Desktop/opencv_baidu/output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_folder, exist_ok=True)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or unable to read.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            raise ValueError("No faces detected.")
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_folder, f"face_{x}_{y}.jpg"), face_img)
    except Exception as e:
        return f"Error in detect_and_crop_faces: {str(e)}"
    return output_folder


def recognize_faces(face_folder, gallery_folder, similarity_threshold=0.97):
    try:
        known_faces = []
        for filename in os.listdir(gallery_folder):
            img_path = os.path.join(gallery_folder, filename)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_faces.append((encodings[0], filename))
        best_matches = {}
        used_gallery_faces = set()  # Set to track which gallery faces have been used

        for face_filename in os.listdir(face_folder):
            face_path = os.path.join(face_folder, face_filename)
            unknown_image = face_recognition.load_image_file(face_path)
            unknown_encodings = face_recognition.face_encodings(unknown_image)
            if unknown_encodings:
                unknown_encoding = unknown_encodings[0]
                face_distances = face_recognition.face_distance([face[0] for face in known_faces], unknown_encoding)
                for i, face_distance in enumerate(face_distances):
                    similarity = np.dot(known_faces[i][0], unknown_encoding) / (np.linalg.norm(known_faces[i][0]) * np.linalg.norm(unknown_encoding))
                    if similarity > similarity_threshold and known_faces[i][1] not in used_gallery_faces:
                        if face_filename not in best_matches or best_matches[face_filename][2] < similarity:
                            # Update the match only if this gallery face has not been used or if this is a better match
                            if face_filename in best_matches:
                                used_gallery_faces.discard(best_matches[face_filename][1])  # Remove old match
                            best_matches[face_filename] = (face_filename, known_faces[i][1], similarity)
                            used_gallery_faces.add(known_faces[i][1])  # Mark this gallery face as used

        return [best_matches[key] for key in best_matches]
    except Exception as e:
        return f"Error in recognize_faces: {str(e)}"


def process(image, gallery_folder):
    try:
        output_folder = detect_and_crop_faces(image.name)
        results = recognize_faces(output_folder, gallery_folder)
        if not results:
            return "No matches found."
        return results
    except Exception as e:
        return f"Error in process function: {str(e)}"

def main():
    gallery_folder = "/Users/xigua/Desktop/opencv_baidu/face"
    image_path = '/Users/xigua/Desktop/opencv_baidu/照片素材/116.jpg'
    # image_path = "/Users/xigua/Desktop/opencv_baidu/识别典型素材/2.jpg"

    class ImageMock:
        def __init__(self, name):
            self.name = name

    image = ImageMock(image_path)
    results = process(image, gallery_folder)

    if results:
        if isinstance(results, list):  # Check if results is a list of tuples
            for result in results:
                if isinstance(result, tuple) and len(result) == 3:
                    print(f"Match found: {result[1]} with similarity {result[2]:.2f} (File: {result[0]})")
                else:
                    print("Error: Result format is incorrect. Received:", result)
        else:
            print("Error:", results)  # Print the error message directly
    else:
        print("No matches found.")

if __name__ == "__main__":
    main()