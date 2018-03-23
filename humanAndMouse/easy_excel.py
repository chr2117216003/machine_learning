#!/usr/bin/env python
# encoding:utf-8
import xlwt

def set_style(name='Times New Roman', bold=False):
    style = xlwt.XFStyle()
    # 设置字体（注意：在同时运行较多文件时，excel字体会报警告）
    # font = xlwt.Font()
    # font.name = name
    # font.bold = bold
    # style.font = font
    # alignment
    alignment = xlwt.Alignment()
    alignment.horz = xlwt.Alignment.HORZ_LEFT
    alignment.vert = xlwt.Alignment.VERT_CENTER
    style.alignment = alignment
    return style

def save(experiment, dimensions, big_results, excel_name):
    try:
        wb = xlwt.Workbook(encoding='utf-8')
        ws = wb.add_sheet(experiment)
        # 写入第一行标题
        row0 = [u'特征集', u'样本个数', u'分类器', u'Accuracy', u'Precision', u'Recall', u'SN', u'SP',
                u'Gm', u'F_measure', u'F_score', u'MCC', u'ROC曲线面积', u'tp', u'fn', u'fp', u'tn']
        for i in range(2, 4):
            ws.col(i).width = 3333 * 2
        for i in range(0, len(row0)):
            ws.write(0, i+1, row0[i], set_style(bold=True))
        # 写入分类结果
        row = 1
        for dimension, results in zip(dimensions, big_results):
            # 合并第一列单元格，写入维度信息
            ws.write_merge(row, row+len(results)-1, 1, 1, dimension+'D', set_style(bold=True))
            # 合并第二列单元格，写入正反例信息
            end = len(results[0])
            note = u'正：'+str(results[0][end-2])+u' 反：'+str(results[0][end-1])
            ws.write_merge(row, row+len(results)-1, 2, 2, note, set_style(bold=True))
            for i in range(0, len(results)):
                for j in range(0, end-2):
                    ws.write(i+row, j+3, results[i][j], set_style())
            row += len(results)
        if excel_name == "":
            excel_name = 'results.xls'
        wb.save(excel_name)
        return True
    except:
        return False
