from models.models import *
from bs4 import BeautifulSoup
from helper import *
import os
import json
import uuid

# CREATE-DROP Tất cả dữ liệu
# db.drop_tables([PDMucLienQuan ,PDTable, PDFile , PDDieu, PDChuong, PDDeMuc, PDChuDe])
# db.create_tables([PDMucLienQuan ,PDTable, PDFile, PDDieu, PDChuong , PDDeMuc, PDChuDe])

# checkpoint = "d8e4a3a0-254c-4593-967c-214ae12bcb0f.html"
checkpoint = None

# Đọc Chủ đề
print("Load Chủ Đề Từ File ...")
with open("./phap-dien/chude.json", "r", encoding="utf-8") as chude_file:
    chudes = json.load(chude_file)
chude_file.close()

print("Insert tất cả chủ đề...")
try:
    with db.atomic():
        PDChuDe.bulk_create([PDChuDe(ten=chude["Text"], stt=chude["STT"], id=chude["Value"]) for chude in chudes])
    print("Inserted tất cả chủ đề pháp điển!")
except:
    pass

# Đọc Đề mục
print("Load Đề Mục Từ File ...")
with open("./phap-dien/demuc.json", "r", encoding="utf-8") as demuc_file:
    demucs = json.load(demuc_file)
demuc_file.close()

print("Insert tất cả chủ đề...")
try:
    with db.atomic():
        PDDeMuc.bulk_create(
            [PDDeMuc(ten=demuc["Text"], stt=demuc["STT"], id=demuc["Value"], chude_id=demuc["ChuDe"]) for demuc in
             demucs])   
except:
    pass
print("Inserted tất cả đề mục pháp điển!")

print("Load Tree Nodes Từ File ...")
with open("./phap-dien/treeNode.json", "r", encoding="utf-8") as tree_nodes_file:
    tree_nodes = json.load(tree_nodes_file)
tree_nodes_file.close()

print("Insert tất cả nodes...")
demuc_directory = os.fsencode("./phap-dien/demuc")
dieus_lienquan = []

count = 0
if checkpoint:
    isSkipping = True
else:
    isSkipping = False
for file in os.listdir(demuc_directory):
    file_name = os.fsdecode(file)
    with open("./phap-dien/demuc/" + file_name, "r", encoding="utf-8") as demuc_file:
        count +=1
        if file_name == checkpoint:
            isSkipping = False
        if isSkipping:
            continue
        demuc_html = demuc_file.read()
        demuc_html = BeautifulSoup(demuc_html, "html.parser")
        demuc_nodes = [node for node in tree_nodes if node["DeMucID"] == file_name.split(".")[0]]
        if len(demuc_nodes) == 0:
            print("Không tìm thấy node cho đề mục: " + file_name)
            demuc_file.close()
            continue
        demuc_chuong = [node for node in demuc_nodes if node["TEN"].startswith("Chương ")]
        chuongs_data = []
        for chuong in demuc_chuong:
            mapc = chuong["MAPC"]
            stt = convert_roman_to_num(chuong["ChiMuc"])
            chuong_data = PDChuong(ten=chuong["TEN"],
                                   mapc=mapc, chimuc=chuong["ChiMuc"],
                                   stt = stt,
                                   demuc_id=chuong["DeMucID"])
            try:
                PDChuong.create(ten=chuong["TEN"],
                                mapc=mapc, chimuc=chuong["ChiMuc"],
                                stt=stt,
                                demuc_id=chuong["DeMucID"])
            except:
                continue
            chuongs_data.append(chuong_data)

        # Insert chương
        print(f'Insert {len(demuc_chuong)} chương của đề mục {file_name}')
        # Tạo một chương giả nếu không có chương
        if len(chuongs_data) == 0:
            chuong_data = PDChuong(ten="",
                                   mapc= uuid.uuid4(), chimuc="0",
                                   stt=0,
                                   demuc_id=file_name.split(".")[0])
            chuongs_data.append(chuong_data)

        demuc_dieus = [node for node in demuc_nodes if node not in demuc_chuong]
        print(f'Đề mục {file_name} có {len(demuc_chuong)} chương và {len(demuc_dieus)} điều')
        stt = 0
        for dieu in demuc_dieus:
            if len(chuongs_data) == 1:
                # Thêm chương giả
                dieu["ChuongID"] = chuongs_data[0].mapc
                # dieu["ChuongID"] = None
            else:
                for chuong in chuongs_data:
                    if dieu["MAPC"].startswith(chuong.mapc):
                        dieu["ChuongID"] = chuong.mapc
                        break

            mapc = dieu["MAPC"]
            dieu_html = demuc_html.select(f'a[name="{mapc}"]')[0]
            ten = dieu_html.nextSibling
            ghi_chu_html = dieu_html.parent.nextSibling
            vbqppl = ghi_chu_html.text if ghi_chu_html else None
            vbqppl_link = ghi_chu_html.select("a")[0]["href"] if ghi_chu_html.select("a") else None
            # print(ten, vbqppl, vbqppl_link)
            noidung_html = dieu_html.parent.find_next("p", {"class": "pNoiDung"})
            noidung = ""
            tables = []
            for content in noidung_html.contents:
                if content.name == "table":
                    tables.append(str(content))
                    continue
                noidung += str(content.text.strip()) + "\n"

            try:
                PDDieu.create(ten=ten, mapc=mapc, chimuc=dieu["ChiMuc"], stt=stt,
                                noidung=noidung, vbqppl=vbqppl, vbqppl_link=vbqppl_link,
                                demuc_id=dieu["DeMucID"], chuong_id=dieu["ChuongID"], chude_id=dieu["ChuDeID"])
            except:
                continue
            for table in tables:
                PDTable.create(dieu_id=mapc, html=table)

            element = noidung_html.nextSibling
            # Lấy link các file, biếu mẫu nếu có đính kèm
            while element and element.name == "a":

                link = element["href"]
                try:
                    PDFile.create(dieu_id=dieu["MAPC"], link=link, path="")
                except:
                    print("Lỗi insert file " + link)

                element = element.nextSibling

            # Lấy các điều có liên quan, nếu có:
            if element and element.name == "p" and element["class"] and element["class"][0] == "pChiDan":
                lienquans_html = element.select("a")
                for lienquan_html in lienquans_html:
                    if not "onclick" in lienquan_html.attrs or lienquan_html["onclick"] == "":
                        continue
                    mapc_lienquan = extract_input(lienquan_html["onclick"]).replace("'", "")
                    dieus_lienquan.append({"dieu_id1": dieu["MAPC"], "dieu_id2": mapc_lienquan})

            stt += 1
        demuc_file.close()

print("Inserted tất cả nodes pháp điển!")

for dieu_lienquan in dieus_lienquan:
    try:
        PDMucLienQuan.create(dieu_id1=dieu_lienquan["dieu_id1"], dieu_id2=dieu_lienquan["dieu_id2"])
    except:
        print(f'Không thể insert liên quan {dieu_lienquan["dieu_id1"]} - {dieu_lienquan["dieu_id2"]}')
    print(f'Inserted liên quan {dieu_lienquan["dieu_id1"]} - {dieu_lienquan["dieu_id2"]}')

