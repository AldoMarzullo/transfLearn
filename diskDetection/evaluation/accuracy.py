#"/home/sciara/Scaricati/TensorBox/data/stare_validation/11_test.jpg": (58.7045454545, 269.5, 90.2045454545, 300.5):0.954303026199, (-10.0227272727, -12.5, 15.75, 12.5):0.00139939249493;
#"data/stare_validation/11_test.jpg": (92.9, 283.1, 59.1, 261.1)
import sys, getopt
from itertools import izip        

def main(argv):

   label_path = ''
   prediction_path = ''
   try:
      opts, args = getopt.getopt(argv,"l:p:",["lfile=","pfile="])
   except getopt.GetoptError:
      print 'test.py -i <labelsfile> -o <predictionfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt in ("-i", "--lfile"):
         label_path = arg
      elif opt in ("-o", "--pfile"):
         prediction_path = arg
   
   success = 0
   samples = 0;
   with open(label_path) as labels, open(prediction_path) as predictions: 
    for x, y in izip(labels, predictions):
        label_box = x.strip().split(":")[1]
        label_box = [float(l) for l in label_box[2:len(label_box)-2].split(",")]
        pred_box = y.strip().split(":")[1]
        pred_box = [float(p) for p in pred_box[2:len(pred_box)-2].split(",")]
        
        #[x1, y1, x2, y2]
	#[x2, y2, x1, y1]
	
	x1_p = x2_l = 0
	x2_p = x1_l = 2
	y1_p = y2_l = 1
	y2_p = y1_l = 3
	
	l1 = Point(label_box[x1_l], label_box[y1_l])
	l2 = Point(label_box[x2_l], label_box[y2_l])
	
	p1 = Point(pred_box[x1_p], pred_box[y1_p])
	p2 = Point(pred_box[x2_p], pred_box[y2_p])

	label_box = Rect(l1,l2)
	pred_box = Rect(p1,p2)
	
	if(overlap(label_box,pred_box)):
	  success += 1
	
	samples += 1

   print "Accuracy: {}%".format(success/samples*100)
  
class Rect(object):
    def __init__(self, p1, p2):
        self.left   = min(p1.x, p2.x)
        self.right  = max(p1.x, p2.x)
        self.bottom = min(p1.y, p2.y)
        self.top    = max(p1.y, p2.y)

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
def overlap(r1, r2):
    return range_overlap(r1.left, r1.right, r2.left, r2.right) and range_overlap(r1.bottom, r1.top, r2.bottom, r2.top)
  
def range_overlap(a_min, a_max, b_min, b_max):
    return not ((a_min > b_max) or (b_min > a_max))
        
if __name__ == "__main__":
   main(sys.argv[1:])