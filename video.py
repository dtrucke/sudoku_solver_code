from concurrent.futures.thread import ThreadPoolExecutor
import threading
import cv2
import imutils
from imutils.perspective import four_point_transform
import numpy as np  
from nn import *
from threading import Thread
from matplotlib import pyplot
from solving import *
#from thread import start_new_thread



# define a video capture object
class  Video(threading.Thread):
    lock = threading.Lock()


    def __init__(self):
        self.model = load_model("LeNet5_modified.h5")
        self.puzzle_array=np.zeros((9,9), dtype=int)
        self.x1 = 0
        self.y1 = 0
        self.max = 0
        self.pred = 0
        self.predict_time = 0
        self.counter = 0
        self.width = 0
        self.hight = 0
        self.y2=0
        self.x2=0
        self.lock = threading.Lock()
        self.puzzle = 0
        self.i = 0
        self.check = False
        self.counter_cut = 0

    def videocap(self, vid):
        counter=0
        counter2 = 0
        while(True):

            
            # Capture the video frame
            # by frame
            ret, frame = vid.read()
            output= frame.copy()

            if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # eliminate noise
            blurred = cv2.GaussianBlur(gray, (7,7), 3)

            # apply adaptive thresholding and then invert the threshold map
            thresh = cv2.adaptiveThreshold(blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
            thresh = cv2.bitwise_not(thresh)
            # Display the resulting frame
        


            cv2.imshow('frame', thresh)

            # find contours    
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

            cnts = imutils.grab_contours(cnts)

            # sort contours 
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            #initialize a contour that corresponds to the puzzle outline
            puzzleCnt = None
            #loop over the contours
            x1 = 0
            y1 = 1

            for c in cnts:
                area = cv2.contourArea(c)
            
                # check if contour is big enough
                if area > 50000:
                    # approximate the contour
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        # if the approximated contour has four points, then we can
                        # assume we have found the outline of the puzzle
                        #
                    counter +=1

                    # check if contour has 4 corners
                    if len(approx) == 4 and len(c)>100:
                        if counter2 == 0:
                         start_timer = time.perf_counter()
                        counter2 += 1

                        puzzleCnt = approx


                        cv2.drawContours(thresh, [puzzleCnt], -1, (0, 255, 0), 2)

                        #cut out Sudoku
                        self.puzzle = four_point_transform(frame, puzzleCnt.reshape(4, 2))
                        t = Thread(target=main.cells, args=(self.puzzle,))
                        width = self.puzzle.shape[1]
                        hight = self.puzzle.shape[0]

                        
                        # start solving
                        if not t.is_alive():
                            t.start()

                        for x in range(9):
                            for y in range(9):
                                cv2.putText(self.puzzle,str(self.puzzle_array[y][x]),
                                    (x*int((width/9)),(y+1)*int((hight/9))), cv2.FONT_HERSHEY_SIMPLEX,2 ,(255, 0, 0),2,cv2.LINE_AA)
                        if self.check:
                            for x in range(9):
                                for y in range(9):
                                    cv2.putText(self.puzzle,str(self.puzzle_array[y][x]),
                                    (x*int((width/9)),(y+1)*int((hight/9))), cv2.FONT_HERSHEY_SIMPLEX,2 ,(255, 0, 0),2,cv2.LINE_AA)

                            end_timer = time.perf_counter()
                            print(f"Gesamte Zeit seit finden des Puzzles: {end_timer-start_timer}")
                        cv2.imshow("Puzzle", self.puzzle)

                        
                        
                    else:
                        cv2.destroyWindow("Puzzle Outline")
                        


                        


            
            # # the 'q' button is set as the
            # # quitting button you may use any
            # # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):

                break


        vid.release()

        cv2.destroyAllWindows()
        return

    # number recognition
    def cells(self,image,debug=False):
        # lock thread 
        self.lock.acquire()
        self.counter = 0
        print("lock aquired")
        print("Thread gestartet")
        if self.counter_cut == 0:
            start_cut=time.perf_counter()
   
        og = image.copy()
       
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        og = image.copy()
        # prepare Sudoku
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        if debug:
            cv2.imshow("Puzzle ", image)
            cv2.waitKey(0)
            cv2.imshow("Puzzle blurred", blurred)
            cv2.waitKey(0)

        # apply adaptive thresholding and then invert the threshold map
        thresh = cv2.adaptiveThreshold(blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 3)
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
        

        # walk through every cell
        for self.x1 in range(9):
            
            for self.y1 in range(9):
                # cut out cells
                if self.y1 > 0 :
                    crop_img =thresh[int((self.y1*y2)-0.1*y2):int(y2*(self.y1+1)+0.1*y2), self.x1*x2:x2*(self.x1+1)]
                    crop_imagetest = og[int((self.y1*y2)-0.1*y2):int(y2*(self.y1+1)+0.1*y2), self.x1*x2:x2*(self.x1+1)]
            
                else:
                    crop_img =thresh[self.y1*y2:y2*(self.y1+1), self.x1*x2:x2*(self.x1+1)]
                    crop_imagetest = og[int((self.y1*y2)-0.1*y2):int(y2*(self.y1+1)+0.1*y2), self.x1*x2:x2*(self.x1+1)]
                
                # cutting of edges
                crop_img=crop_img[self.y1+int(0.1*y2):y2+int(0.1*y2), self.x1+int(0.1*x2):x2-int(0.1*x2)] #relativer Rand
                crop_img2 = crop_img.copy()
                #crop_imagetest = crop_imagetest[self.y1+int(0.1*y2):y2+int(0.1*y2), self.x1+int(0.1*x2):x2-int(0.1*x2)] #relativer Rand
                




                cnts = cv2.findContours(crop_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
                width2 = crop_img.shape[1]
                hight2 = crop_img.shape[0]
                # check for numbers
                if cnts:
                    for c in cnts:
                        M = cv2.moments(c)
                        # get center of contour
                        if M['m00'] != 0:
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])	
                        
                        else:
                            cx = 0
                            cy = 0
                        x,y,w,h = cv2.boundingRect(c)
                        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (36,255,12), 2)
                        # cout out number 
                        roi=crop_img2[y-2:y+h+2,x-2:x+w+2]

                        roi2 = roi
           
               
                        # center of contour in the middle predict number
                        if cx > 0.25*width2 and cx < 0.75*width2 and h>w and len(c) >= 15:

                            roi= cv2.resize(roi,dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

                            roi = roi.reshape( 1,28, 28, 1)
                            roi = roi.astype('float32')
                            roi/=255.0
                            

                            # predict number
                            cell=main.test_fkt1(roi)

                            self.puzzle_array[self.y1][self.x1]=cell

                            # if number is 7 and the contour is shorter than 32 set number to 1 
                            # not optimal depends on the size of the input
                            if cell == 7 and len(c) <32:
                                self.puzzle_array[self.y1][self.x1]=1                          
                            break
               

                        # if np contou in the middle cell =0
                        self.puzzle_array[self.y1][self.x1]=0
                else:    
                    # if no contour in the image cell = 0
                    self.puzzle_array[self.y1][self.x1]=0



                font = cv2.FONT_HERSHEY_SIMPLEX
        font = cv2.FONT_HERSHEY_SIMPLEX
        end_cut = time.perf_counter()  

        #solve sudoku      
        if (solveSuduko(self.puzzle_array, 0, 0)):
            self.check = True

   
        # if no solution take next frame
        else:
            
            self.i +=1
            self.lock.release()
            return False

        print("lock released")            
        return True
    

    # Destroy all the windows
    cv2.destroyAllWindows()

    # predict number
    def test_fkt1(self,test_img):

            start_predict=time.perf_counter()
            self.pred = self.model.predict(test_img)
     
            test=np.array(self.pred[0])
            self.max=self.pred.argmax()
            # if np.max(test)>0.97:
            if self.max == 0:
                cell = 6
            
            else:
              
                cell=self.max
  
            end_preditct = time.perf_counter()
            self.predict_time += end_preditct - start_predict
            self.counter +=1
            print(f"Predict dauert: {end_preditct-start_predict}")
            return cell

    cv2.destroyAllWindows()

    def get_video(self):
        return



if __name__ == "__main__":
    test()    
    vid = cv2.VideoCapture("test_videos/sudoku_leicht_pc.mp4")
    while vid.isOpened():
        main = Video()
        main.videocap(vid)


