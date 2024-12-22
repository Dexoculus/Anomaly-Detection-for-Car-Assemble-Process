import os
import win32com.client as win32

def convert_xlxs(file_path):
    """
    Convert .xls format files to .xlsx format.
    Delete Original Files.
    """
    excel = win32.gencache.EnsureDispatch('Excel.Application')
    wb = excel.Workbooks.Open(file_path)
    new_file_path = file_path + "x" # .xlsx file name
    try:
        wb.SaveAs(new_file_path, FileFormat=51) # FileFormat=51: .xlsx
        wb.Close()
        excel.Application.Quit()
        print(f"Converted: {file_path} to {new_file_path}")
        os.remove(file_path) # Delete Original File
        print(f"Deleted original file: {file_path}")
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        wb.Close()
        excel.Application.Quit()

def convert_all_xls_in_folders(base_dir):
    """
    Converting all .xls files in the base_dir directory and subdirectories.
    Delete Original Files after conversion.
    """
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.xls') and not file.endswith('.xlsx'): # filter only .xls files
                file_path = os.path.join(root, file) # Generate file path
                try:
                    convert_xlxs(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

def main():
    base_directory = os.path.abspath("./data/robot_action") # Root Directory for search
    print(f"Starting conversion in directory: {base_directory}")
    convert_all_xls_in_folders(base_directory)
    print("Conversion completed.")

if __name__ == "__main__":
    main()