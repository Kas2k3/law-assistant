from db import db
from peewee import *


class LongTextField(TextField):
    """MySQL LONGTEXT (4GB) — TextField mặc định là TEXT (giới hạn 64KB),
    không đủ cho HTML bảng lớn trong một số điều luật."""
    field_type = "LONGTEXT"


class BaseModel(Model):
    class Meta:
        database = db


class PDChuDe(BaseModel):
    id = CharField(max_length=128, primary_key=True)
    ten = TextField()
    stt = IntegerField()


class PDDeMuc(BaseModel):
    id = CharField(max_length=128, primary_key=True)
    ten = TextField()
    stt = IntegerField()
    chude_id = ForeignKeyField(PDChuDe, backref="demucs")


class PDChuong(BaseModel):
    mapc = CharField(max_length=128, primary_key=True)
    ten = TextField()
    demuc_id = ForeignKeyField(PDDeMuc, backref="chuongs")
    chimuc = TextField()
    stt = IntegerField()


class PDDieu(BaseModel):
    ten = TextField()
    demuc_id = ForeignKeyField(PDDeMuc, backref="dieus")
    chuong_id = ForeignKeyField(PDChuong, backref="dieus")
    # chuong_id = ForeignKeyField(PDChuong, backref="dieus", null=True, on_delete='SET NULL')
    chude_id = ForeignKeyField(PDChuDe, backref="dieus")
    mapc = CharField(max_length=128, primary_key=True)
    noidung = LongTextField()
    chimuc = IntegerField()
    vbqppl = TextField()
    vbqppl_link = TextField(null=True)
    stt = IntegerField()


class PDTable(BaseModel):
    dieu_id = ForeignKeyField(PDDieu, backref="tables")
    html = LongTextField()


class PDFile(BaseModel):
    dieu_id = ForeignKeyField(PDDieu, backref="files")
    link = TextField()
    path = TextField()


class PDMucLienQuan(BaseModel):
    dieu_id1 = ForeignKeyField(PDDieu)
    dieu_id2 = ForeignKeyField(PDDieu)


# db.create_tables([PDDieu, PDChuong , PDDeMuc, PDChuDe, PDTable])
db.create_tables([PDChuDe, PDDeMuc, PDChuong, PDDieu, PDTable, PDFile, PDMucLienQuan])
