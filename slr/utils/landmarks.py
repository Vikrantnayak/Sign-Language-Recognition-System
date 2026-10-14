import cv2 as cv


def draw_landmarks(image, landmark_point):
    
    magenta = (255, 0, 255)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    neon_green = (20, 255, 57)
    neon_purple = (253, 18, 171)
    neon_blue = (255, 81, 31)
    neon_orange = (31, 91, 255)
    neon_red = (49, 49, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)

    skeletal_color = white
    outline_color = black
    
    if len(landmark_point) > 0:

        skeletal_color = neon_green
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), outline_color, 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), outline_color, 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), skeletal_color, 2)

        skeletal_color = neon_purple
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), outline_color, 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), outline_color, 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), outline_color, 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), skeletal_color, 2)

        skeletal_color = neon_red
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), outline_color, 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), outline_color, 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), outline_color, 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), skeletal_color, 2)

        skeletal_color = neon_blue
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), outline_color, 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), outline_color, 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), outline_color, 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), skeletal_color, 2)

        skeletal_color = neon_orange
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), outline_color, 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), outline_color, 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), outline_color, 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), skeletal_color, 2)
        
        skeletal_color = yellow
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), outline_color, 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), outline_color, 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), outline_color, 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), outline_color, 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), outline_color, 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), outline_color, 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), outline_color, 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), skeletal_color, 2)
        
        skeletal_color = white

    for index, landmark in enumerate(landmark_point):
        
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 1:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 2:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 3:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 4:  
            cv.circle(image, (landmark[0], landmark[1]), 8, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, outline_color, 1)
        
        if index == 5:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 6:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 7:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 8:  
            cv.circle(image, (landmark[0], landmark[1]), 8, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, outline_color, 1)
        
        if index == 9:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 10:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 11:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 12:  
            cv.circle(image, (landmark[0], landmark[1]), 8, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, outline_color, 1)
        
        if index == 13:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 14:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 15:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 16:  
            cv.circle(image, (landmark[0], landmark[1]), 8, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, outline_color, 1)
        
        if index == 17:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 18:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 19:  
            cv.circle(image, (landmark[0], landmark[1]), 5, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, outline_color, 1)
        
        if index == 20:  
            cv.circle(image, (landmark[0], landmark[1]), 8, skeletal_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, outline_color, 1)

    return image