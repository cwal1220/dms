import cv2
import dlib

LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

# 눈 크기 측정 함수
def get_eye_size(landmarks):
    # 눈 위치 찾기
    left_eye = [(landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y)] # 37 41
    right_eye = [(landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(45).x, landmarks.part(45).y)] # 43 47

    # 눈 가로 길이 측정
    left_eye_width = left_eye[1][0] - left_eye[0][0]
    right_eye_width = right_eye[1][0] - right_eye[0][0]
    eye_width = (left_eye_width + right_eye_width) // 2

    # 눈 세로 길이 측정
    left_eye_height = (left_eye[1][1] - left_eye[0][1])
    right_eye_height = (right_eye[1][1] - right_eye[0][1])
    eye_height = (left_eye_height + right_eye_height) // 2

    return eye_width, eye_height

def draw_landmarks(landmarks, frame):
    # 눈의 위치에 점 그리기
    for index in LEFT_EYE_POINTS:
        cv2.circle(frame, (landmarks.part(index).x, landmarks.part(index).y), 2, (0, 255, 0), -1)
    for index in RIGHT_EYE_POINTS:
        cv2.circle(frame, (landmarks.part(index).x, landmarks.part(index).y), 2, (0, 255, 0), -1)

    return frame

# 비디오 캡처 객체 초기화
cap = cv2.VideoCapture(0)

# dlib의 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    # 이미지 읽기
    ret, image = cap.read()
    
    # 얼굴 검출
    faces = detector(image, 1)

    # 랜드마크 검출 및 눈 크기 측정
    for face in faces:
        landmarks = predictor(image, face)

        # 눈 크기 측정
        eye_width, eye_height = get_eye_size(landmarks)

        # 눈 크기 출력
        cv2.putText(image, f'Eye width: {eye_width}px', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f'Eye height: {eye_height}px', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 눈 위치 표시
        # left_eye = [(landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y)]
        # right_eye = [(landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(45).x, landmarks.part(45).y)]
        # cv2.rectangle(image, left_eye[0], left_eye[1], (0, 255, 0), 2)
        # cv2.rectangle(image, right_eye[0], right_eye[1], (0, 255, 0), 2)
        draw_landmarks(landmarks, image)

    # 이미지 출력
    cv2.imshow('Image', image)

    # q키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
