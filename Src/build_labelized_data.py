# -*- coding: utf-8 -*-
"""
@author: AnicetSIÉWÉ KOUÉTA: 20172760
@author: Benjamin BERGÉ: 20182608
@author: Jolan WATHELET: 20182628

conda install -c anaconda urllib3

"""
import os
import re
import itertools
import csv
import socket

import re
import json
import time


OUTPUT_PATH = "../Output/Supervized"
DATA_PATH = "../SupervizedTrainingData"

GOOD_CSV_PATH = [DATA_PATH+'/Good/domcop_top10milliondomains.csv']
BAD_CSV_PATH = [DATA_PATH+'/Bad/Prof/malwaredomainlist_off_2017_90726.csv']
GOOD_TXT_PATH = [] #None - pas de fonctions comme on aura prob pas besoin
BAD_TXT_PATH = [DATA_PATH+'/Bad/Prof/zeus_dga_domains_dec_2014_31000.txt', 
                DATA_PATH+'/Bad/Phishing/openfish_2021_1135.txt',
                DATA_PATH+'/Bad/Mix/Zonefiles.io_2021_114017.txt']
TRAINING_DATASET_PATH = OUTPUT_PATH+'/training_dataset.csv'
IP_COUNTRY_CSV_PATH = OUTPUT_PATH+'/ip_country_trainig_dataset.csv'

domain_regex = r'(([\da-zA-Z])([_\w-]{,62})\.){,127}(([\da-zA-Z])[_\w-]{,61})?([\da-zA-Z]\.((xn\-\-[a-zA-Z\d]+)|([a-zA-Z\d]{2,})))'
domain_regex2 = '{0}$'.format(domain_regex)
valid_domain_name_regex = re.compile(domain_regex2, re.IGNORECASE)

def strip_registar_name(domain):
    return re.sub('.[a-z]+$', '', domain) 

def verify_input_paths():
    for filename in itertools.chain(GOOD_CSV_PATH,BAD_CSV_PATH,GOOD_TXT_PATH,BAD_TXT_PATH): 
        if not os.path.exists(filename): # dest path doesnot exist
            print ("ERROR: input file from the web expected but not provided: \n filepath:", filename)
            return -1
            
def is_a_domain_name(string):
    if re.match(valid_domain_name_regex, string ):
        return True
    else:
        return False
    

"""
use a specific csv format, see code
"""
def read_bad_csv(training_csv_writer):
    evil_dom_counter = 0
    for csv_path in BAD_CSV_PATH:
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    domain = row[1]
                    reviplookup = row[3]
                    if is_a_domain_name(domain):
                        domain = strip_registar_name(domain)
                        training_csv_writer.writerow([domain,1])
                        evil_dom_counter += 1
                    if is_a_domain_name(reviplookup):
                        reviplookup = strip_registar_name(reviplookup)
                        training_csv_writer.writerow([reviplookup,1])
                        evil_dom_counter += 1
                line_count += 1
    return evil_dom_counter
                
"""
one domain name per row
"""
def read_bad_txt(training_csv_writer):
    evil_dom_counter = 0
    for p in BAD_TXT_PATH:
        f = open(p, 'r')
        lines = f.readlines()
        for l in lines:
            if is_a_domain_name(l):
                domain = strip_registar_name(l[:-1])
                training_csv_writer.writerow([domain,1])
                evil_dom_counter += 1
    return evil_dom_counter

    
"""
use a specific csv format, see code
"""
def read_good_csv(training_csv_writer, number_to_read):
    number_written = 0
    for csv_path in GOOD_CSV_PATH:
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    domain = row[1]
                    if is_a_domain_name(domain):
                        domain = strip_registar_name(domain)
                        training_csv_writer.writerow([domain,0])
                        number_written += 1
                if number_written == number_to_read:
                    return
                line_count += 1
    
        

def build_csv_dataset():
    verify_input_paths()
    with open(TRAINING_DATASET_PATH, mode='w', newline='') as training_file:
        training_csv_writer = csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        evil_domain_count = read_bad_csv(training_csv_writer)
        evil_domain_count += read_bad_txt(training_csv_writer)
        print(evil_domain_count," evil domain names were written")
        read_good_csv(training_csv_writer, evil_domain_count)
        print(evil_domain_count," good domain names were written")
        print("for a total of ", (evil_domain_count*2) )
        print("Training CSV is finished")
        
#--------------- Second dataset (Domain name, IP, country) ------------------------#
        
def read_ip_country_bad_csv(training_csv_writer):
    evil_dom_counter = 0
    for csv_path in BAD_CSV_PATH:
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    domain = row[1]
                    ip = row[2]
                    country = row[8]
                    if is_a_domain_name(domain):
                        domain = strip_registar_name(domain)
                        training_csv_writer.writerow([domain,ip,country,1])
                        evil_dom_counter += 1
                    
                line_count += 1
    return evil_dom_counter 

def query_good_ip(training_csv_writer,number_to_read):
    number_written = 0
    for csv_path in GOOD_CSV_PATH:
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    domain = row[1]
                    if is_a_domain_name(domain):
                        ip = socket.gethostbyname(domain)
                        time.sleep(2)
                        country = '/'
                        domain = strip_registar_name(domain)
                        training_csv_writer.writerow([domain,ip,country,0])
                        number_written += 1
                if number_written == number_to_read:
                    return
                line_count += 1

def ip_country_dataset():
    verify_input_paths()
    with open(IP_COUNTRY_CSV_PATH, mode='w', newline='') as training_file:
        training_csv_writer = csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        evil_domain_count = read_ip_country_bad_csv(training_csv_writer)
        print("evil ", evil_domain_count) #4835
        query_good_ip(training_csv_writer,evil_domain_count)
        
    

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    build_csv_dataset()
    #ip_country_dataset()
    
