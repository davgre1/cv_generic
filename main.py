import cv2
import qoi
import easyocr
import pytesseract
import pandas as pd
import numpy as np
from PIL import Image
from time import time
from pytesseract import Output
from PyPDF2 import PdfFileReader, PdfFileWriter
from math import dist


def image_to_bytes(img_ext):
    img = cv2.imread("./cv_receipt_generic_v2/incoming_images/tj3.jpg")
    img_bytes = cv2.imencode(img_ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tobytes() # Saves image as byte string
    return img_bytes
    

def bytes_to_image(img_ext, img_bytes):
    img_nparr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("./cv_receipt_generic_v2/final_images/byte_img" + img_ext, img_nparr)


def image_compression(img_ext):
    img = cv2.imread("./cv_receipt_generic_v2/incoming_images/tj3.jpg")
    img_bytes = cv2.imencode(img_ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tobytes()
    img_nparr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    _ = qoi.write("./cv_receipt_generic_v2/final_images/img.qoi", img_nparr)
    rgb_read = qoi.read("./cv_receipt_generic_v2/final_images/img.qoi")
    cv2.imshow("rgb_read", rgb_read)
    cv2.waitKey(0)


def image_to_pdf():
    img = Image.open("./cv_receipt_generic_v2/incoming_images/tj3.jpg")
    img1 = img.convert('RGB')
    img1.save("./cv_receipt_generic_v2/final_images/tj3.pdf")


def create_pdf():
    # https://www.blog.pythonlibrary.org/2018/05/29/creating-interactive-pdf-forms-in-reportlab-with-python/
    # https://stackabuse.com/creating-a-form-in-a-pdf-document-in-python-with-borb/
    
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfform
    from reportlab.lib.colors import magenta, pink, blue, green

    c = canvas.Canvas('./cv_receipt_generic_v2/final_images/simple_form.pdf')
    
    c.setFont("Courier", 20)
    c.drawCentredString(750, 1061, 'Employment Form')
    c.setFont("Courier", 14)
    form = c.acroForm
    
    c.drawString(10, 635, '')
    form.textfield(name='fname', tooltip='First Name', x=120, y=640, borderStyle='inset', width=310, height=25, forceBorder=True)
    
    c.drawString(10, 605, '')
    form.textfield(name='lname', tooltip='Last Name', x=120, y=605, borderStyle='inset', width=310, height=25, forceBorder=True)
    
    c.drawString(10, 585, '')
    form.textfield(name='address', tooltip='Address', x=120, y=570, borderStyle='inset', width=310, height=25, forceBorder=True)
    c.save()


def image_to_edit_pdf():
    out = pytesseract.image_to_pdf_or_hocr(Image.open("./cv_receipt_generic_v2/incoming_images/eng_invoice2.png"), config=r'-l eng --oem 1 --psm 11 hocr')
    f = open("./cv_receipt_generic_v2/final_images/eng_invoice2.pdf", "w+b")
    f.write(bytearray(out))
    f.close()

    # Overlay
    with open("./cv_receipt_generic_v2/final_images/simple_form.pdf", "rb") as overlay, open("./cv_receipt_generic_v2/final_images/eng_invoice2.pdf", "rb") as  inFile:
        original = PdfFileReader(inFile)
        background = original.getPage(0)
        foreground = PdfFileReader(overlay).getPage(0)
        background.mergePage(foreground)

        writer = PdfFileWriter()
        for i in range(original.getNumPages()):
            page = original.getPage(i)
            writer.addPage(page)
        with open("./cv_receipt_generic_v2/final_images/test.pdf", "wb") as outFile:
            writer.write(outFile)


def image_vectorization_one():
    image = cv2.imread("./cv_receipt_generic_v2/incoming_images/tj3.jpg") # F74Yq.png tj3.jpg invoice_test

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # COORDS
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect

    maxWidth = max(int(dist(br, bl)), int(dist(tr, tl)))
    maxHeight = max(int(dist(tr, br)), int(dist(tl, bl)))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    (thresh, warped) = cv2.threshold(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)

    cv2.imwrite('./cv_receipt_generic_v2/final_images/cropped_image.jpg', warped)


def image_vectorization_many():
    image = cv2.imread("./cv_receipt_generic_v2/incoming_images/invoice_test.png") # F74Yq.png tj3.jpg invoice_test

    inputCopy = image.copy()

    # CREATING NEW CANVAS
    white_img = np.zeros([inputCopy.shape[0], inputCopy.shape[1], 3], dtype=np.uint8)
    white_img.fill(255)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    
    d = pytesseract.image_to_data(image, lang='eng', config=r'-l eng --oem 1 --psm 6', output_type=Output.DICT)
    df = pd.DataFrame(d)
    # df = df.head(25)
    df = df[df.block_num != '0'] 
    df = df[df.top != 0].reset_index()
    
    df['height_mean'] = df['height'].groupby(df['line_num'], group_keys=False).transform('mean').astype(int)
    df['top_mean'] = df['top'].groupby(df['line_num'], group_keys=False).transform('mean').astype(int)
    
    # cv2.getTextSize(text, font, font_scale, thickness)
    d = df.to_dict('dict')
    for level in range(len(d['level'])):
        if float(d['conf'][level]) > 30:
            n = 5 # increase box size
            (x1, y1, x2, y2) = (d['left'][level], d['top'][level], d['width'][level], d['height'][level])
            cropped_img = inputCopy[int(y1-n):int(y1 + y2)+n, int(x1-n):int(x1 + x2)+n]
            cv2.imwrite('./cv_receipt_generic_v2/test/img_temp.png', cropped_img)
                    
            data = pytesseract.image_to_string(cropped_img, config=' -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#@$/\-(). --psm 7', output_type=Output.DICT)
            if data != '':
                string_data = data['text'].replace("\n", " ").replace("\x0c", "")
                print(string_data)
            y2_mean = d['height_mean'][level]
            y1_mean = d['top_mean'][level]
            # cropped_img = cv2.rectangle(white_img, ((x1), (y1_mean)), ((x1 + x2), (y1_mean + y2_mean)), (0, 255, 0), 2)
            cropped_img2 = cv2.putText(white_img, string_data, (x1, y1_mean+30), cv2.FONT_HERSHEY_DUPLEX, (y2_mean/40), (0, 0, 0), 2)
    cv2.imwrite('./cv_receipt_generic_v2/final_images/scanned_image3.png', cropped_img)


def image_vectorization_many_easyocr():
    image = cv2.imread("./cv_receipt_generic_v2/final_images/cropped_image.jpg") # F74Yq.png tj3.jpg invoice_test
    inputCopy = image.copy()

    # CREATING NEW CANVAS
    white_img = np.zeros([inputCopy.shape[0], inputCopy.shape[1], 3], dtype=np.uint8)
    white_img.fill(255)
    
    reader = easyocr.Reader(['en'])
    detection_result = reader.detect(image, width_ths=0.7, mag_ratio=1.5)
    text_coordinates = detection_result[0][0]
    data = reader.recognize(image, horizontal_list=text_coordinates, free_list=[])

    for dd in data:
        string_data = dd[1]
        x1 = dd[0][0][0]
        y1 = dd[0][0][1]
        x2 = dd[0][2][0]
        y2 = dd[0][2][1]
        # cropped_img = cv2.rectangle(white_img, ((x1), (y1)), ((x2), (y2)), (0, 255, 0), 2)
        cropped_img = cv2.putText(white_img, string_data, (x1+40, y2), cv2.FONT_HERSHEY_DUPLEX, abs(y2-y1)/70, (0, 0, 0), 2)
    cv2.imwrite('./cv_receipt_generic_v2/final_images/scanned_image3.png', cropped_img)
    

def image_to_text_layout():
    d = pytesseract.image_to_data(Image.open("./cv_receipt_generic/incoming_images/IMG_1292.jpg"), config=r'-l eng --oem 3 --psm 6', output_type=Output.DICT)
    df = pd.DataFrame(d)
    df = df.head(55)
    # print(df)

    # out = pytesseract.image_to_pdf_or_hocr(Image.open("./cv_receipt_generic_v2/incoming_images/eng_invoice.png"), config=r'-l eng --oem 1 --psm 11 hocr')
    # f = open("./cv_receipt_generic_v2/final_images/demofile.pdf", "w+b")
    # f.write(bytearray(out))
    # f.close()
    
    df.rename(columns = {'left':'x', 'top':'y', 'width':'w', 'height':'h'}, inplace = True) # 'width':'w', 'height':'h'
    df = df[df.block_num != '0'] 
    df['text'] = df.apply(lambda x: '\n'  if x['conf'] == '-1' else x['text'], axis=1) # NEW LINES

    # df3 = df.groupby('line_num', as_index=False).last().assign(text='\n')
    # df = pd.concat([df, df3]).sort_values(['line_num'], kind='stable', ignore_index=True)
    # df.drop(['level', 'page_num', 'word_num', 'par_num'], axis=1, inplace=True)
    
    df['x2'] = df[['w', 'x']].sum(axis=1)
    df['y2'] = df[['y', 'h']].sum(axis=1)
    df1 = df
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    
    space_to_pixel = 20 # pixel/space
    df1['spaces'] = (df1['x'].shift(-1) - df1['x2']).shift(1).fillna(min(df1['x'])).astype(int)
    df1['spaces'] = df1.apply(lambda x: str(int(x['spaces']/space_to_pixel)*' ')  if x['spaces'] != 0 else str(int((x['y'])/space_to_pixel)*' '), axis=1) # SPACES BETWEEN BLOCKS
    df1['text'] = [' '.join(i) for i in zip(df1['spaces'], df1['text'])] # LEFT EVERY WORD FRONT SPACES
    df1 = df1.groupby(['line_num', 'block_num'], as_index=False).agg({'x':'min', 'y':'mean', 'x2':'max', 'y2':'mean', 'text': lambda x: ''.join(x)}).round(2) # SPACES BETWEEN WORDS
    df1['txt'] = df1.apply(lambda x: ''*int(x['x']/space_to_pixel), axis=1) # LEFT-END SPACES
    df1['text'] = [''.join(i) for i in zip(df1['txt'], df1['text'])] # LEFT FRONT SPACES

    curr = df1[df1['block_num'] == 1]
    text = ''

    for ix, ln in curr.iterrows():
        text += ln['text']
    print(text)


def remove_text_from_image():
    pass


def generic_dilate_img():
    image = cv2.imread("./cv_receipt_generic_v2/final_images/cropped_image.jpg") # F74Yq.png tj3.jpg invoice_test
    inputCopy = image.copy()
    grayInput = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binaryImage = cv2.threshold(grayInput, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('', binaryImage)
    cv2.waitKey(0)
    kernelSize = (2, 2)
    opIterations = 3 # increase box
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    dilateImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    cv2.imshow('', dilateImage)
    cv2.waitKey(0)
    

def rebuild_image():
    image = cv2.imread("./cv_receipt_generic_v2/final_images/cropped_image.jpg") # F74Yq.png tj3.jpg invoice_test
    inputCopy = image.copy()

    # CREATING NEW CANVAS
    white_img = np.zeros([inputCopy.shape[0], inputCopy.shape[1], 3], dtype=np.uint8)
    white_img.fill(255)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    
    d = pytesseract.image_to_data(image, lang='eng', config=r'-l eng --oem 1 --psm 6', output_type=Output.DICT)
    df = pd.DataFrame(d)
    # df = df.head(25)
    df = df[df.block_num != '0'] 
    df = df[df.top != 0].reset_index()
    
    df['height_mean'] = df['height'].groupby(df['line_num'], group_keys=False).transform('mean').astype(int)
    df['top_mean'] = df['top'].groupby(df['line_num'], group_keys=False).transform('mean').astype(int)
    
    # cv2.getTextSize(text, font, font_scale, thickness)
    d = df.to_dict('dict')
    for level in range(len(d['level'])):
        if float(d['conf'][level]) > 30:
            n = 5 # increase box size
            (x1, y1, x2, y2) = (d['left'][level], d['top'][level], d['width'][level], d['height'][level])
            # print((d['height_mean'][level])/40)
            cropped_img = inputCopy[int(y1-n):int(y1 + y2)+n, int(x1-n):int(x1 + x2)+n]
            # EXPENSIVE
            data = pytesseract.image_to_string(cropped_img, config=' -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#@$/\-(). --psm 7', output_type=Output.DICT)
            if data != '':
                string_data = data['text'].replace("\n", " ").replace("\x0c", "")
                print(string_data)
            y2_mean = d['height_mean'][level]
            y1_mean = d['top_mean'][level]
            # cropped_img = cv2.rectangle(white_img, ((x1), (y1_mean)), ((x1 + x2), (y1_mean + y2_mean)), (0, 255, 0), 2)
            cropped_img = cv2.putText(white_img, string_data, (x1, y1_mean+30), cv2.FONT_HERSHEY_DUPLEX, (y2_mean/40), (0, 0, 0), 2)
    cv2.imwrite('./cv_receipt_generic_v2/final_images/scanned_image3.png', cropped_img)


def image_cropper():
    image = cv2.imread("./cv_receipt_generic_v2/incoming_images/tj3.jpg") # F74Yq.png tj3.jpg invoice_test

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # COORDS
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect

    maxWidth = max(int(dist(br, bl)), int(dist(tr, tl)))
    maxHeight = max(int(dist(tr, br)), int(dist(tl, bl)))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    (thresh, warped) = cv2.threshold(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), 185, 255, cv2.THRESH_BINARY)
    
    # scale_percent = 60 
    # width = int(warped.shape[1] * scale_percent / 100)
    # height = int(warped.shape[0] * scale_percent / 100)
    # resized = cv2.resize(warped, (width, height), interpolation = cv2.INTER_AREA)
    cv2.imwrite('./cv_receipt_generic_v2/final_images/cropped_image.jpg', warped)


if __name__ == "__main__":
    start_time = time()
    image_to_text_layout()
    print("--- %s sec ---" % (time()-start_time))