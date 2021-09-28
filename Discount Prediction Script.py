# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:13:52 2021

@author: YC
"""

from flask import Flask, request, render_template
import numpy as np
import pickle
from pickle import load
import pandas as pd
pd.set_option('display.float_format', '{:.6f}'.format)
import flasgger
from flasgger import Swagger
from tensorflow import keras
from keras.models import load_model
import tensorflow
app=Flask(__name__)
Swagger(app)

## Loading Trained model and Minmax Transformation Pickle files

regression_model = tensorflow.keras.models.load_model('D:\Discount Prediction - Machine Learning\Discount Prediction - ANN & Machine Learning\Discount_ANN_MinMax.h5')
minmax_independent = load(open("D:\Discount Prediction - Machine Learning\Discount Prediction - ANN & Machine Learning\min_max_independent.pkl","rb"))
minmax_dependent = load(open("D:\Discount Prediction - Machine Learning\Discount Prediction - ANN & Machine Learning\min_max_dependent.pkl","rb"))

@app.route('/', methods=['GET'])
def welcome():
    return render_template('index.html')

@app.route('/predict',methods=["Get"])
def predict_discount():
    
    ## Fetching values 
    insured_value=float(request.args.get("insured_value"))
    lead_type_random=request.args.get("lead_type")  
    policy_price=float(request.args.get("policy_price"))
    car_make=request.args.get("car_make").strip()
    car_type=request.args.get("car_type")
    coverage_purchased=request.args.get("coverage_purchased")
    provider_name=request.args.get("provider_name")
    product_name=request.args.get("product_name")
    is_takaful=request.args.get("is_takaful")
    # Day need to be passed without preceding zero
    paid_day=int(request.args.get("paid_day"))
    # Month need to be passed without preceding zero
    paid_month=int(request.args.get("paid_month"))
    paid_year=int(request.args.get("paid_year"))
    excess_mean=float(request.args.get("excess"))
    nationality_random=request.args.get("nationality")
    roadside_assistance_random=request.args.get("roadside_assistance")
    
    ## Null, Empty, Space
    # Same value is given to LOST and ACTIVE_LOST. Values are assigned as per lead_type average discount value
    # Higher value means more discount provided to that lead type
    # Need to pass upper case value 
    lead_type_random = lead_type_random.upper()
    lead_type_dict = {'ACTIVE_LOST':4, 'LOST':4, 'RENEWAL':3, 'NORMAL':2, 'RENEWAL_LOST':1}
    if lead_type_random in lead_type_dict:
        lead_type_random = lead_type_dict[lead_type_random]
    else:
        lead_type_random = 1
    
    # Capitalize the first letter of every word
    car_make = car_make.title()
    car_make_dict = {'Borgward': 1, 'Ssangyong': 2, 'Maxus': 3, 'Isuzu': 4, 'Aston Martin': 5, 'Lada': 6, 'Genesis': 7, 
                     'Luxgen': 8, 'Brilliance': 9, 'McLaren': 10, 'Lotus': 11, 'Saab': 12, 'Rolls Royce': 13, 'RAM': 14, 
                     'Ferrari': 15, 'Lamborghini': 16, 'Haval': 17, 'Citroen': 18, 'GAC': 19, 'Tesla': 20, 'Alfa Romeo': 21,
                     'Bentley': 22, 'Fiat': 23, 'Seat': 24, 'Jac': 25, 'Mercury': 26, 'Changan': 27, 'Opel': 28, 
                     'Chery': 29, 'Maserati': 30, 'Hummer': 31, 'Daihatsu': 32, 'Skoda': 33, 'Chrysler': 34, 'Geely': 35, 
                     'Subaru': 36, 'MG': 37, 'Mini': 38, 'Cadillac': 39, 'Lincoln': 40, 'Volvo': 41, 'Jaguar': 42, 
                     'Peugeot': 43, 'Porsche': 44, 'GMC': 45, 'Suzuki': 46, 'Landrover': 47, 'Dodge': 48, 'Renault': 49, 
                     'Jeep': 50, 'Audi': 51, 'Infiniti': 52, 'Lexus': 53, 'Mercedes': 54, 'BMW': 55, 'Mazda ': 56, 
                     'Volkswagen': 57, 'Chevrolet': 58, 'Kia': 59, 'Hyundai': 60, 'Ford': 61, 'Mitsubishi': 62, 
                     'Honda': 63, 'Toyota': 64, 'Nissan': 65}
    if car_make in car_make_dict:
        car_make = car_make_dict[car_make]
    else:
        car_make = 1
    
    # converts the first character of the string to a capital
    # 4x4
    car_type = car_type.capitalize()
    car_type_dict = {'Pick-up': 1, 'VAN': 2, 'Convertible': 3, 'Coupe': 4, '4x4': 5, 'Sedan': 6}
    if car_type in car_type_dict:
        car_type = car_type_dict[car_type]
    else:
        car_type = 1
    
    # values will be 'Comprehensive' and 'Third Party'
    # Capitalize the first letter of every word
    coverage_purchased = coverage_purchased.title()
    coverage_purchased_dict = {'Comprehensive':1,'Third Party':0}
    if coverage_purchased in coverage_purchased_dict:
        coverage_purchased = coverage_purchased_dict[coverage_purchased]
    else:
        coverage_purchased = 1
    
    
    # Need to pass provider name as it is
    provider_name_dict = {'Al Fujairah National Insurance Co': 1, 'Abu Dhabi National Insurance Company': 2, 
                          'Al Sagr Insurance': 3, 'i-Insured Insurance': 4, 'Al-Ittihad Al Watani': 5, 'RSA Insurance': 6, 
                          'Al Hilal Takaful PSC': 7, 'Dubai Insurance': 8, 'Oman Insurance': 9, 
                          'The New India Assurance Co. Ltd.': 10, 'Dar Al Takaful Insurance': 11, 
                          'Al Buhaira National Insurance': 12, 'Al Wathba Insurance': 13, 
                          'Dubai Islamic Ins. and Reins. AMAN': 14, 'Methaq Takaful Insurance': 15, 
                          'Abu Dhabi National Takaful': 16, 'Oriental Insurance Company': 17, 'Watania Insurance': 18, 
                          'Insurance House': 19, 'Union Insurance': 20, 'Dubai National Insurance and Reinsurance': 21, 
                          'Adamjee Insurance': 22, 'Tokio Marine and Nichido Fire Insurance Co. Ltd.': 23, 
                          'Salama Insurance': 24, 'Noor Takaful Insurance': 25}
    
    if provider_name in provider_name_dict:
        provider_name = provider_name_dict[provider_name]
    else:
        provider_name = 1
     
    # Need to pass product name as it is    
    product_name_dict = {'Premium Garage Plus': 1, 'Gold Agency': 1, 'TPL': 2, 'Exclusive Dyna Trade': 3, 'Prestige': 4, 
                         'Silver Agency': 5, 'Almasa Plus': 6, 'Premium Garages': 7, 'Dynatrade': 8, 'Premium Plus': 9, 
                         'Platinum': 10, 'PLUS - Dyna Trade/Premium Garage': 11, 'Premium Garage': 12, 'Third Party': 13, 
                         'Elite': 14, 'Smart Comprehensive': 15, 'ASAS': 16, 'Edge': 17, 'Comprehensive IV': 18, 
                         'Basic': 19, 'Mumtaz With Car Hire': 20, 'Silver': 21, 'Gold': 22, 'Comprehensive': 23, 
                         'Premium': 24, 'Mumtaz': 25, 'Must': 26, 'Standard': 27, 'Third Party Only': 28}
    
    if product_name in product_name_dict:
        product_name = product_name_dict[product_name]
    else:
        product_name = 1
        
    ## Need to pass only True and False
    is_takaful_dict = {False:0,True:1}
    
    if is_takaful in is_takaful_dict:
        is_takaful = is_takaful_dict[is_takaful]
    else:
        is_takaful = 1
        
    # Capitalize the first letter of every word    
    nationality_random_dict = {'Andorran':1,'Rwandan':1,'Solomon Islands':1, 'Namibian': 1, 'Yugoslav': 2, 'Guyanese': 3,
                               'Bahamian': 4, 'Laotian': 5, 'Fijian': 6, 'Bruneian': 7, 'Guatemalan': 8, 'Swazi': 9,
                               'Angolan': 10, 'Salvadorean': 11, 'Sierra Leonian': 12, 'Barbadian': 13, 'Djiboutian': 14,
                               'Luxembourg': 15, 'Grenadian': 16, 'Ecuadorean': 17, 'Liberian': 18, 'Nigerien': 19,
                               'Beninese': 20, 'Seychellois': 21, 'Paraguayan': 22, 'Burkinese': 23, 'Congolese': 24,
                               'Bolivian': 25, 'Surinamese': 26, 'Taiwanese': 27, 'Gabonese': 28, 'Trinidadian/Tobagonian': 29,
                               'Cuban': 30, 'Maldivian': 31, 'Chadian': 32, 'Panamanian': 33, 'Belizian': 34, 'Qatari': 35,
                               'Vanuatuan': 36, 'Costa Rican': 37, 'Georgian': 38, 'Montenegrin': 39, 'Burundian': 40,
                               'Icelandic': 41, 'Guinean': 42, 'Gambian': 43, 'Botswanan': 44, 'Estonian': 45,
                               'Maltese': 46, 'Hong Kong': 47, 'Chilean': 48, 'Kittitians and Nevisians': 49, 'Zambian': 50,
                               'Kuwaiti': 51, 'Norwegian': 52, 'Cypriot': 53, 'Vietnamese': 54, 'Slovenian': 55,
                               'Jamaican': 56, 'Mauritanian': 57, 'Burmese': 58, 'Argentinian': 59, 'Latvian': 60,
                               'Senegalese': 61, 'Scottish': 62, 'Albanian': 63, 'Finnish': 64, 'Eritrean': 65,
                               'Ghanaian': 66, 'Lithuanian': 67, 'Bosnian': 68, 'Japanese': 69, 'Czech': 70, 'Ugandan': 71,
                               'Libyan': 72, 'Croatian': 73, 'Azerbaijani': 74, 'Thai': 75, 'Swiss': 76, 'Tajik': 77,
                               'Mauritian': 78, 'Mexican': 79, 'Dominican': 80, 'Slovak': 81, 'Venezuelan': 82,
                               'Austrian': 83, 'Macedonian': 84, 'Cameroonian': 85, 'Moldovan': 86, 'Bulgarian': 87,
                               'Danish': 88, 'Colombian': 89, 'Turkmen': 90, 'Hungarian': 91, 'Singaporean': 92,
                               'Belgian': 93, 'Belorussian': 94, 'Tanzanian': 95, 'Swedish': 96, 'Kyrgyz': 97,
                               'Bahraini': 98, 'Somali': 99, 'New Zealand': 100, 'Armenian': 101, 'Afghan': 102,
                               'Ethiopian': 103, 'Indonesian': 104, 'Zimbabwean': 105, 'Omani': 106, 'Polish': 107,
                               'Comorian': 108, 'Kazakh': 109, 'Nigerian': 110, 'Greek': 111, 'Dutch': 112, 'Portuguese': 113,
                               'Brazilian': 114, 'South Korean': 115, 'Serbian': 116, 'Saudi Arabian': 117, 'Spanish': 118,
                               'Malaysian': 119, 'Kenyan': 120, 'German': 121, 'Italian': 122, 'Ukrainian': 123,
                               'Nepalese': 124, 'Romanian': 125, 'Uzbek': 126, 'Algerian': 127, 'Irish': 128, 'Chinese': 129,
                               'Turkish': 130, 'Russian': 131, 'Tunisian': 132, 'Australian': 133, 'Yemeni': 134,
                               'Moroccan': 135, 'French': 136, 'Bangladeshi': 137, 'Iranian': 138, 'Iraqi': 139, 'South African': 140,
                               'Canadian': 141, 'American': 142, 'Sri Lankan': 143, 'Sudanese': 144, 'Palestini': 145, 'Lebanese': 146,
                               'British': 147, 'Syrian': 148, 'Philippine': 149, 'Jordanian': 150, 'Emirati': 151, 'Egyptian': 152,
                               'Pakistani': 153, 'Indian': 154}
    
    if nationality_random in nationality_random_dict:
        nationality_random = nationality_random_dict[nationality_random]
    else:
        nationality_random = 1
        
    # Passed value should not be null
    roadside_assistance_random_dict = {'platinum': 1, '50': 2, '20': 3, 'none': 4, '30': 5, '10': 6, 'silver': 7, 'gold': 8}
    
    if roadside_assistance_random in roadside_assistance_random_dict:
        roadside_assistance_random = roadside_assistance_random_dict[roadside_assistance_random]
    else:
        roadside_assistance_random = 1
    
    # Creating dictionary to finally create a data frame and feed the value to trained model 
    data = {'insured_value':[insured_value],
        'lead_type_random': [lead_type_random],
        'policy_price':[policy_price],
        'car_make':[car_make],
        'car_type':[car_type],
        'coverage_purchased':[coverage_purchased],
        'provider_name':[provider_name],
        'product_name':[product_name],
        'is_takaful':[is_takaful],
        'paid_day':[paid_day],
        'paid_month':[paid_month],
        'paid_year':[paid_year],
        'excess_mean':[excess_mean],
        'nationality_random':[nationality_random],
        'roadside_assistance_random':[roadside_assistance_random]}
        
    df = pd.DataFrame(data)
    X_train_independent_minmax=minmax_independent.transform(df)
    X_train_independent_minmax_final = np.ravel(X_train_independent_minmax).tolist()
    prediction=regression_model.predict([X_train_independent_minmax_final])
    predicted_values = minmax_dependent.inverse_transform(prediction)
    print(predicted_values)

        


if __name__=='__main__':
    app.run(debug=True)