from colorama import init, Fore, Style
init(autoreset=True)
def intro():
    print(
        f"{Fore.CYAN}╭─────────────────────────────────────────────────────────────────╮{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}                                                                 {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}   {Style.BRIGHT}{Fore.BLUE}██████╗  ██████╗  ██████╗██╗   ██╗██████╗  █████╗{Style.RESET_ALL}             {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}   {Style.BRIGHT}{Fore.BLUE}██╔══██╗██╔═══██╗██╔════╝██║   ██║██╔══██╗██╔══██╗{Style.RESET_ALL}            {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}   {Style.BRIGHT}{Fore.BLUE}██║  ██║██║   ██║██║     ██║   ██║██████╔╝███████║{Style.RESET_ALL}            {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}   {Style.BRIGHT}{Fore.BLUE}██║  ██║██║   ██║██║     ██║   ██║██╔══██╗██╔══██║{Style.RESET_ALL}            {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}   {Style.BRIGHT}{Fore.BLUE}██████╔╝╚██████╔╝╚██████╗╚██████╔╝██║  ██║██║  ██║{Style.RESET_ALL}            {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}   {Style.BRIGHT}{Fore.BLUE}╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝{Style.RESET_ALL}            {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}                                                                 {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}   {Style.BRIGHT}{Fore.YELLOW}Smart Document Processing & Analysis Tool{Style.RESET_ALL}                     {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}                                                                 {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}├─────────────────────────────────────────────────────────────────┤{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}                                                                 {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}   {Style.BRIGHT}{Fore.GREEN}Welcome to Docura v1.0.0{Style.RESET_ALL}                                      {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}   Transform, analyze, and process documents with intelligence   {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}                                                                 {Fore.CYAN}│{Style.RESET_ALL}\n"
#        f"{Fore.CYAN}│{Style.RESET_ALL}   {Style.BRIGHT}Quick Start:{Style.RESET_ALL}                                                  {Fore.CYAN}│{Style.RESET_ALL}\n"
#        f"{Fore.CYAN}│{Style.RESET_ALL}   {Fore.MAGENTA}•{Style.RESET_ALL} {Fore.WHITE}docura analyze document.pdf{Style.RESET_ALL}                                 {Fore.CYAN}│{Style.RESET_ALL}\n"
#        f"{Fore.CYAN}│{Style.RESET_ALL}   {Fore.MAGENTA}•{Style.RESET_ALL} {Fore.WHITE}docura convert file.docx --format markdown{Style.RESET_ALL}                  {Fore.CYAN}│{Style.RESET_ALL}\n"
#        f"{Fore.CYAN}│{Style.RESET_ALL}   {Fore.MAGENTA}•{Style.RESET_ALL} {Fore.WHITE}docura extract data.pdf --type tables{Style.RESET_ALL}                       {Fore.CYAN}│{Style.RESET_ALL}\n"
#        f"{Fore.CYAN}│{Style.RESET_ALL}                                                                 {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}   Visit '{Fore.LIGHTCYAN_EX}/{Style.RESET_ALL}' endpoint for information                            {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}   Visit {Fore.LIGHTCYAN_EX}https://tinyurl.com/docura-ai{Style.RESET_ALL} for documentation         {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}│{Style.RESET_ALL}                                                                 {Fore.CYAN}│{Style.RESET_ALL}\n"
        f"{Fore.CYAN}╰─────────────────────────────────────────────────────────────────╯{Style.RESET_ALL}"
    )
intro()