import picoweb
from machine import SoftI2C, Pin
from lcd_api import LcdApi
from i2c_lcd import I2cLcd

ssid = 'wifiHP'
password = '1234567955898'

ip = ''

i2c = SoftI2C(scl=Pin(22), sda=Pin(21), freq=10000)
lcd = I2cLcd(i2c, 0x20, 2, 16)


def do_connect(ssid,password):
    global ip
    import network
    sta_if = network.WLAN(network.STA_IF)
    if not sta_if.isconnected():
        print('connecting to network...')
        sta_if.active(True)
        sta_if.connect(ssid, password)
        while not sta_if.isconnected():
            pass
    print('network config:', sta_if.ifconfig())
    ip = sta_if.ifconfig()[0]

def qs_parse(qs):
 
  parameters = {}
  ampersandSplit = qs.split("&")
 
  for element in ampersandSplit:
    equalSplit = element.split("=")
    parameters[equalSplit[0]] = equalSplit[1]
 
  return parameters


do_connect(ssid,password)

app = picoweb.WebApp(__name__)


@app.route("/")
def index(req, resp):
    queryString = (req.qs).replace("%20", " ")
    parameters = qs_parse(queryString)
    yield from picoweb.start_response(resp)
    
    #add program here
    lcd.clear()
    lcd.putstr(parameters['data'])
    #print(parameters['data'])

app.run(debug=True, host = ip ,port=8080)