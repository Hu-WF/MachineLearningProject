#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-22
# Version:     2.1.0
# E-mail:      674649741@qq.com
# Purpose:     Ploting
#==============================================================================
import mlFunctions as f
import data_information as di

def main():
    pt=f.MLPloting()
    pt.plot_back_and_all_samples()
    pt.plot_mean_samples()
#    pt.plot_PCA(6)
    pt.plot_PCA(di.global_class_num)
    return 0

if __name__=='__main__':
    main()
    