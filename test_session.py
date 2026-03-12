from utils.session_manager import SessionManager

# create manager
session = SessionManager()

# add dataset (dummy)
session.add_dataset("sample1", {"dummy": "dataset"})

print("Datasets:", session.list_datasets())

# add result
session.add_result("sample1", "two_band", {"chi2": 1.2})

print("Results:", session.get_results())

# clear results
session.clear_results()

print("Results after clear:", session.get_results())