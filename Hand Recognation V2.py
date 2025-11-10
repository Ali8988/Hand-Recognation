import cv2
import mediapipe as mp

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# نقاط رؤوس الأصابع
tipids = [4, 8, 12, 16, 20]

# قاموس بسيط (عدد الأصابع ↔ كلمة)
word_dict = {
    1: "Hi",
    2: "Yes",
    3: "No",    
    4: "Okay",
    5: "Thanks"
}

# لتتبع الحركة (إشارات ديناميكية)
prev_x, prev_y = 0, 0
movement_text = ""

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    Lmlist = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, Lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(Lm.x * w), int(Lm.y * h)
                Lmlist.append([id, cx, cy])

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if len(Lmlist) == 21:
                fingers = []

                # الإبهام
                if Lmlist[tipids[0]][1] < Lmlist[tipids[0] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # باقي الأصابع
                for tip in range(1, 5):
                    if Lmlist[tipids[tip]][2] < Lmlist[tipids[tip] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                totalfingers = fingers.count(1)

                # عرض العدد
                cv2.putText(img, f'{totalfingers}', (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)

                # عرض الكلمة المرتبطة
                if totalfingers in word_dict:
                    cv2.putText(img, word_dict[totalfingers], (200, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

                # تتبع حركة اليد (Dynamic Gestures)
                cx, cy = Lmlist[9][1], Lmlist[9][2]  # مركز اليد (عند المعصم تقريبا)

                if prev_x != 0 and prev_y != 0:
                    dx = cx - prev_x
                    dy = cy - prev_y

                    if abs(dx) > 30:  # حركة يمين أو شمال
                        if dx > 0:
                            movement_text = "Swipe Right → Hello"
                        else:
                            movement_text = "Swipe Left → Goodbye"

                    elif abs(dy) > 30:  # حركة فوق أو تحت
                        if dy > 0:
                            movement_text = "Swipe Down → Stop"
                        else:
                            movement_text = "Swipe Up → Go"

                prev_x, prev_y = cx, cy

                if movement_text != "":
                    cv2.putText(img, movement_text, (100, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('Hand Recognition + Sign Language', img)

    if cv2.waitKey(25) & 0xff == 27:  # زر Esc للخروج
        break

cap.release()
cv2.destroyAllWindows()
