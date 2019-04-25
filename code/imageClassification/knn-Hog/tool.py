# -*- coding:utf8 -*-
import os
path = '../images/rubbish/'
filelist = os.listdir(path)
for item in filelist:

        if item.endswith('.jpg'):
                temp = item.split(".")
                # name = item.split('.',3)[0] + '.' + item.split('.',3)[1]
                src = os.path.join(os.path.abspath(path),item)
                dst = os.path.join(os.path.abspath(path),"2_"+temp[0] + '.jpg')

        try:
            os.rename(src, dst)
            print('rename from %s to %s' % (src, dst))
        except:
            continue

