from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

service = Service(ChromeDriverManager().install())

options = Options()
options.add_experimental_option('detach', True) # this ensures window does not close

chrome_browser = webdriver.Chrome(service = service, options = options)
chrome_browser.get('https://www.qaplayground.com/practice')

chrome_browser.maximize_window()

sth = chrome_browser.find_element(By.CLASS_NAME, 'gradient-subTitle')
print(sth.get_attribute('innerHTML'))

assert 'QA PlayGround' in chrome_browser.page_source

# updated code for button
# button = chrome_browser.find_element(By.CLASS_NAME, 'btn-default')