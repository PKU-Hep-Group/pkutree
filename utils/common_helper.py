import time
import string
import random
import os
import data
# The absolute path
# This function takes as input any path (relativate path to pkutree/data), and returns the absolute path
def abs_path(path_in_repo):
    return os.path.join(data.__path__[0], path_in_repo)

# [   END   ]
# []

def cout(mode="test", info=None, **kwargs):
    if len(kwargs) == 0:
        newargs = ""
    else:
        newargs = kwargs

    if mode == "start":
        try:
            print("[  START  ]",f">>> {info} <<<" if info != None else "","entries:", kwargs['entries'])
        except:
            print("[  START  ]",f">>> {info} <<<" if info != None else "", newargs)
        return time.time()
    elif mode == "end":
        if "entries" in kwargs.keys():
            try:
                print("[   END   ]",f">>> {info} <<<" if info != None else "",\
                    "entries:",kwargs['entries'],", cost time:",np.round(time.time()-kwargs['start_time'],2),"s")
                return
            except:
                pass
        if "start_time" in kwargs.keys():
            try:
                print("[   END   ]",f">>> {info} <<<" if info != None else "","cost time:",np.round(time.time()-kwargs['start_time'],2),"s")
                return
            except:
                pass
        print("[   END   ]",f">>> {info} <<<" if info != None else "", newargs)
        return

    elif mode == "summary":
        print("[ SUMMARY ]",f">>> {info} <<<" if info != None else "", newargs)
    elif mode == "warning":        
        print("[ WARNING ]",f">>> {info} <<<" if info != None else "", newargs)
    elif mode == "error":        
        print("[  ERROR  ]",f">>> {info} <<<" if info != None else "", newargs)
    else:
        print("[   OUT   ]", mode)
    
    return


def get_randomstr(length=11):
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, length))
    return ran_str