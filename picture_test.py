from this import d
import cv2
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
from nn import *
from solving import *
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
from threading import Thread


class Picture_C:

    def __init__(self):
        self.model = load_model("saved_model_annealer_45_2.h5")
        self.puzzle_array=np.zeros((9,9), dtype=int)
        self.x1 = 0
        self.y1 = 0
        self.max = 0
        self.pred = 0
        self.predict_time = 0
        self.counter = 0
        self.h =[]
        self.w=[]

        

    def find_puzzle(self,image, debug=False):
        #blur it slightly
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)

        # apply adaptive thresholding and then invert the threshold map
        thresh = cv2.adaptiveThreshold(blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh)

        # check to see if we are visualizing each step of the image
        # processing pipeline (in this case, thresholding)
        if debug:
            cv2.imshow("Puzzle Thresh", thresh)
            cv2.waitKey(0)
        # find contours in the thresholded image and sort them by size in
        # descending order
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # initialize a contour that corresponds to the puzzle outline
        puzzleCnt = None
        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we can
            # assume we have found the outline of the puzzle
            if len(approx) == 4:
                puzzleCnt = approx
                #print(puzzleCnt)
                break
            # if the puzzle contour is empty then our script could not find
        # the outline of the Sudoku puzzle so raise an error
        if puzzleCnt is None:
            raise Exception(("Could not find Sudoku puzzle outline. "
                "Try debugging your thresholding and contour steps."))
        # check to see if we are visualizing the outline of the detected
        # Sudoku puzzle
        if debug:
            # draw the contour of the puzzle on the image and then display
            # it to our screen for visualization/debugging purposes
            output = image.copy()
            cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
            cv2.imshow("Puzzle Outline", output)
            cv2.imwrite("puttle_outline.jpg", output)
            cv2.waitKey(0)
                # apply a four point perspective transform to both the original
        # image and grayscale image to obtain a top-down bird's eye view
        # of the puzzle
        puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
        warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
        cv2.imwrite("fourpint.jpg",puzzle)
        #puzzle = cv2.imread("puzzle.jpg")
        gray2 = cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)
        # check to see if we are visualizing the perspective transform
        return (puzzle, warped,gray2)

    def cells(self,image,debug=False):
        start_for=time.perf_counter()
   
        og = image.copy()
        blurred = cv2.GaussianBlur(image, (7, 7), 0)
        if debug:
            cv2.imshow("Puzzle ", image)
            cv2.waitKey(0)
            cv2.imshow("Puzzle blurred", blurred)
            cv2.waitKey(0)
        # apply adaptive thresholding and then invert the threshold map
        thresh = cv2.adaptiveThreshold(blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        thresh = cv2.bitwise_not(thresh)
        # check to see if we are visualizing each step of the image
        # processing pipeline (in this case, thresholding)
        
        if debug:
            cv2.imshow("Puzzle Thresh", thresh)
            cv2.waitKey(0)
        width = image.shape[1]
        hight = image.shape[0]
        y2=int(hight/9)
        x2=int(width/9)
        
        
        counter = 0
        cutting_with_number = 0
        for self.x1 in range(9):
            
            for self.y1 in range(9):
                start_cutting = time.perf_counter()
                if self.y1 > 0 :
                    crop_img =thresh[int((self.y1*y2)-0.1*y2):int(y2*(self.y1+1)+0.1*y2), self.x1*x2:x2*(self.x1+1)]
            
                else:
                    crop_img =thresh[self.y1*y2:y2*(self.y1+1), self.x1*x2:x2*(self.x1+1)]


                
                # cutting of edges
                crop_img=crop_img[self.y1+int(0.1*y2):y2+int(0.1*y2), self.x1+int(0.1*x2):x2-int(0.1*x2)] #relativer Rand
                crop_img2 = crop_img.copy()

                            


                cnts = cv2.findContours(crop_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
                width2 = crop_img.shape[1]
                hight2 = crop_img.shape[0]
                # check for numbers

                if cnts:
                    for c in cnts:
                        #c = cnts[0]
                        M = cv2.moments(c)
                        if M['m00'] != 0:
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])	
                        else:
                            cx = 0
                            cy = 0
                        x,y,w,h = cv2.boundingRect(c)
                        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (36,255,12), 2)
                        roi=crop_img2[y-2:y+h+2,x-2:x+w+2]
                        roi3=crop_img2[y:y+h,x:x+w]
                        
                        #roi = crop_img2
                        roi2 = roi
                        #roi = crop_img2
                        # roi = roi3
                     
                        
                        roi= cv2.resize(crop_img,dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
                        roi = roi.reshape( 1,28, 28, 1)
                        roi = roi.astype('float32')
                        roi/=255.0

               
                        if cx > 0.25*width2 and cx < 0.75*width2 and h>w and len(c) > 20:
                           
                            end_cutting_with_number = time.perf_counter()
                            counter +=1
                            cutting_with_number += end_cutting_with_number-start_cutting

                                               
                            cell=main.test_fkt1(roi)
                            # if cell == 8:
                            #     cv2.imshow("8",roi2)
                            #     cv2.waitKey()

                            self.puzzle_array[self.y1][self.x1]=cell
                                                   
                            break
                   


                        self.puzzle_array[self.y1][self.x1]=0
                else:    
                    self.puzzle_array[self.y1][self.x1]=0



                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(og,str(self.puzzle_array[self.y1][self.x1]),((self.x1)*x2,(self.y1+1)*y2),font,2 ,(255, 0, 0),2,cv2.LINE_AA)
        # cv2.imshow("thresh",thresh)
        cv2.imshow(file,og)
        cv2.waitKey()
        end_for=time.perf_counter()
        #print(f"Es wurden {counter} Nummern gefunden jedes Auschneiden dauert: {cutting_with_number/counter} Sekunden")
        print(f"Prediction Time:{end_for-start_for}")

                    
        return

    def test_fkt1(self,test_img):

        start_predict=time.perf_counter()
        self.pred = self.model.predict(test_img)

        #print(pred)
        test=np.array(self.pred[0])
        self.max=self.pred.argmax()
        # if np.max(test)>0.97:
        if self.max == 0:
            cell = 1
        
        else:
            cell=self.max

     
            

        end_preditct = time.perf_counter()
        self.predict_time += end_preditct - start_predict
        self.counter +=1
        return cell

    cv2.destroyAllWindows()

if __name__ == "__main__":
    #test()
    result = np.array([[8, 0, 0, 0, 1, 0, 0, 0, 9],
        [0, 5 ,0 ,8, 0, 7, 0, 1, 0,],
        [0, 0, 4, 0, 9, 0, 7, 0, 0],
        [0, 6, 0, 7, 0, 1, 0, 2, 0],
        [5, 0, 8, 0, 6, 0, 1, 0, 7],
        [0, 1, 0, 5, 0, 2, 0, 9, 0],
        [0, 0, 7, 0, 4, 0, 6, 0, 0],
        [0, 8, 0, 3, 0, 9, 0, 4, 0],
        [3, 0, 0, 0, 5, 0, 0, 0, 8]])
    lis =[]
    lis.append(np.array([[0,6,0,0,0,0,0,7,0],
        [5,0,7,0,0,0,2,0,4],
        [0,8,0,5,0,4,0,6,0],
        [0,0,5,9,0,1,7,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,6,2,0,3,9,0,0],
        [0,7,0,6,0,2,0,3,0],
        [3,0,4,0,0,0,8,0,6],
        [0,2,0,0,0,0,0,5,0]]))

    lis.append(np.array([[0,0,2,0,3,0,6,0,0],
            [0,0,0,4,0,5,0,0,0],
            [8,0,4,0,0,0,7,0,2],
            [0,2,0,0,0,0,0,8,0],
            [3,0,0,0,0,0,0,0,6],
            [0,6,0,0,0,0,0,1,0],
            [5,0,8,0,0,0,1,0,7],
            [0,0,0,7,0,3,0,0,0],
            [0,0,1,0,6,0,4,0,0]]))

    lis.append(np.array([[0,5,3,4,0,7,6,1,0],
            [8,0,0,0,0,0,0,0,3],
            [0,4,0,0,0,0,0,9,0],
            [3,0,0,8,0,6,0,0,5],
            [0,0,0,0,0,0,0,0,0],
            [6,0,0,5,0,9,0,0,8],
            [0,1,0,0,0,0,0,2,0],
            [2,0,0,0,0,0,0,0,4],
            [0,8,9,6,0,3,7,5,0]]))

    lis.append(np.array([[0,3,0,1,4,8,2,0,0],
            [0,4,2,0,0,0,7,0,0],
            [9,0,0,0,7,0,5,0,4],
            [2,9,8,0,6,0,0,7,0],
            [0,5,0,0,0,2,0,4,6],
            [0,0,0,3,8,0,0,5,0],
            [0,0,5,8,2,0,6,0,0],
            [0,2,0,0,9,0,3,8,0],
            [0,6,9,7,0,0,0,2,0]]))

    lis.append(np.array([[0,0,1,0,0,0,0,0,3],
            [4,0,6,0,7,0,2,0,0],
            [9,3,0,0,1,0,0,0,0],
            [3,1,0,2,0,0,0,6,0],
            [5,0,0,0,6,8,0,0,0],
            [0,0,0,0,0,0,0,2,8],
            [0,0,4,8,0,6,0,0,2],
            [6,0,7,9,0,3,8,0,4],
            [8,2,3,1,4,0,9,0,0]]))

    lis.append(np.array([[4,0,0,8,0,2,0,0,6],
            [0,6,0,0,1,0,0,4,0],
            [0,3,8,0,0,0,1,7,0],
            [0,0,0,1,2,7,0,0,0],
            [7,0,0,3,0,4,0,0,1],
            [0,0,0,5,8,9,0,0,0],
            [0,7,4,0,0,0,2,1,0],
            [0,2,0,0,3,0,0,6,0],
            [9,0,0,2,0,8,0,0,5]]))

    lis.append( np.array([[0,0,5,8,0,4,0,0,0],
            [0,3,0,1,0,0,7,6,0],
            [0,1,0,5,0,0,0,0,4],
            [5,0,0,0,0,0,4,1,6],
            [0,0,0,0,0,0,0,0,0],
            [1,6,2,0,0,0,0,0,9],
            [4,0,0,0,0,8,0,5,0],
            [0,5,3,0,0,7,0,2,0],
            [0,0,0,3,0,9,1,0,0]]))

    lis.append( np.array([[2,0,8,0,0,9,7,0,4],
            [0,0,6,0,0,7,1,0,0],
            [0,3,0,0,0,4,0,2,0],
            [8,7,2,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,9,5,1],
            [0,2,0,1,0,0,0,4,0],
            [0,0,4,7,0,0,6,0,0],
            [6,0,1,9,0,0,3,0,8]]))

    lis.append(  np.array([[9,0,6,0,0,0,3,0,8],
            [0,0,0,0,7,0,0,0,0],
            [0,0,3,0,0,0,5,0,0],
            [1,0,0,8,0,2,0,0,3],
            [0,0,0,0,5,0,0,0,0],
            [2,0,0,7,0,6,0,0,4],
            [0,0,5,0,0,0,8,0,0],
            [0,0,0,0,2,0,0,0,0],
            [6,0,8,0,0,0,4,0,1]]))

    lis.append(np.array([[0,0,0,0,1,0,0,0,0],
            [0,0,0,5,0,3,0,0,0],
            [0,2,3,7,4,9,5,1,0],
            [9,3,0,0,0,0,0,2,5],
            [0,4,1,0,6,0,3,9,0],
            [7,5,0,0,0,0,0,6,4],
            [0,8,9,2,3,6,7,4,0],
            [0,0,0,1,0,4,0,0,0],
            [0,0,0,0,8,0,0,0,0]]))
    check = False
    i=0
    genau = []
    w=[]
    h=[]
    laufzeit =[]
    prediction =0
    pred_time =0
    zahlen = 0
    while i <= 9:
        u = i+1
        result = lis[i]
        file = 'test_images/00'+str(u)+".jpg"
        if i == 9:
            file = 'test_images/0'+str(u)+".jpg"

        #file = "002.jpg"
        #print(file)
        start_all = time.perf_counter()
        main = Picture_C()
        img = cv2.imread(file)
        width = img.shape[1]
        hight = img.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX


    
        width = img.shape[1]
        hight = img.shape[0]
        start=time.perf_counter()
        str1 = ""

        puzzle, wraped,gray2= main.find_puzzle(img)
        ende=time.perf_counter()
        width = int(gray2.shape[1]/9)
        hight = int(gray2.shape[0]/9)
        #find_cell(puzzle)
        w.append(width)
        h.append(hight)
        start_sud = time.perf_counter()
        main.cells(gray2)
        print(f"test{main.puzzle_array}")
        predict = np.zeros((9,9), dtype=int)
        predict = main.puzzle_array.copy()
        start_solving = time.perf_counter()
        # if (solveSuduko(main.puzzle_array, 0, 0)):
        #     print(main.puzzle_array)

        #     #printing(main.puzzle_array)
        #     for x in range(9):
        #         for y in range(9):
        #             cv2.putText(gray2,str(main.puzzle_array[y][x]),(x*width,(y+1)*hight),font,2 ,(255, 0, 0),2,cv2.LINE_AA)

        #     check = False
        #     end_all = time.perf_counter()
        # else:
        #     print("no solution exists ")
        # end_all = time.perf_counter()

        # end_solving = time.perf_counter()
        # print(f"Find_puzzle dauert:{ende-start}")

        #print(f"Lösen dauert: {end_solving-start_solving}")
        print(f"Jede Prediction dauert {main.predict_time/main.counter}")
        pred_time +=main.predict_time
        prediction +=main.predict_time/main.counter
        #print(f"Rest:{end_all-start_all}")
        print(main.counter)
        genauigkeit =0
        for x in range(9):
            for y in range(9):
                if result[x][y] == predict[x][y] and predict[x][y]!=0:
                    genauigkeit +=1
        print(result)
        print(predict)
        print(genauigkeit)
        print(float(genauigkeit/main.counter))
        genau.append(float(genauigkeit/main.counter))
        zahlen +=main.counter

        i+=1
        if check == True:
            cv2.imshow("Solved", gray2)
            cv2.waitKey()
        end_sud = time.perf_counter()
        laufzeit.append(end_sud-start_sud)
    print(genau)
    print(h)
    print(w)

    
    av = 0
    l=0
    for ele in laufzeit:
        l+=ele
    for ele in genau:
        av+=ele
    print(f"av: {av/10}")
    print(f"laufzeit:{l/10}")
    print(f"jede pred:{prediction/10}")
    print(f" pred:{pred_time/10}")    
    print(zahlen/10)
