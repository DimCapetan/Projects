import requests
import hashlib

def request_api_data(char):
    url = 'https://api.pwnedpasswords.com/range/' + char
    res = requests.get(url)
    
    if res.status_code != 200:
        raise RuntimeError(f'Error fetching: {res.status_code}, check API and try again!')
    
    return res

def get_password_leaks_count(hashes, hash_to_check):
    hashes = (line.split(':') for line in hashes.text.splitlines())
    for h, count in hashes:
        if h == hash_to_check:
            return count
    return 0

def pwned_api_check(password):
    pswrd = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
    first5_char, tail = pswrd[:5], pswrd[5:]
    response = request_api_data(first5_char)
    return get_password_leaks_count(response, tail)

def main(*args):
    for password in args:
        count = pwned_api_check(password)
        if count:
            print(f'{password} was found {count} time in a breach. You should probably change your password!')
        else:
            print(f'{password} was not found. Carry on!')
    return 'Done!'

d = ['1', '2!', '3', '4!']

if __name__ == '__main__':
    main(*d)

