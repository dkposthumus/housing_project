from striprtf.striprtf import rtf_to_text
import numpy as np
from pathlib import Path
import re
import json

# === Set up paths ===
home = Path.home()
work_dir = home / "housing_project"
data = work_dir / "data"
minutes = data / "meeting_minutes"
minutes_raw = minutes / "raw"
minutes_clean = minutes / "processed"
text_dir = minutes / "text"
text_dir.mkdir(parents=True, exist_ok=True)

rtf = Path(f"{minutes_raw}/minutes_training_sample.rtf").read_text(encoding="utf-8", errors="ignore")
plain = rtf_to_text(rtf)
# now `plain` is the same text you see, with all the <<Project Start>> markers
blocks = re.findall(r"<<Project Start>>(.*?)<<Project End>>", plain, flags=re.DOTALL)

# create a dictionary which are the labels for the training sample
labels = [
    {
        'case_number': '98.226D',
        'project_address': '571 Jersey Street',
        'lot_number': '033',
        'assessor_block': '6540',
        'project_descr': 'Request for Discretionary Review of Building Permit Application No. 9722606 to construct a third-level bedroom suite and roof deck to an existing two-story single-family residence',
        'type_district': 'RH-2',
        'type_district_descr': 'house, two-family',
        'speakers': 'None',
        'action': 'Continued as proposed',
        'ayes': ['Chinchilla', 'Theoharis', 'Antenore', 
            'Hills', 'Joe', 'Martin', 'Mills'],
        'noes': [],
        'vote': '7-0',
    },
    {
        'case_number': '98.251C',
        'project_address': '154 Coleridge Street',
        'lot_number': '22',
        'assessor_block': '5642',
        'project_descr': 'Request for authorization of a conditional use to modify a condition of approval of Commission Motion No. 11774 to accommodate one handicapped-accessible dwelling unit in subject three-unit residential building',
        'type_district': 'RH-2',
        'type_district_descr': 'house, two-family',
        'height_and_bulk_district': '40-x',
        'action': 'Continued as proposed',
        'speakers': 'None',
        'ayes': ['Chinchilla', 'Theoharis', 'Antenore', 
            'Hills', 'Joe', 'Martin', 'Mills'],        
        'noes': [],
        'vote': '7-0'
    },
    {
        'case_number': '97.686C',
        'project_address': '1100 Grant Avenue',
        'lot_number': '10',
        'assessor_block': '162',
        'project_descr': 'Request for Conditional Use authorization under Sections 812.49 and 812.20 of the Planning Code to establish a financial service of approximately 3,600 feet in the CR-NC (Chinatown Residential Neighborhood Commercial) District',
        'height_and_bulk_district': '50-N',
        'action': 'Continued as proposed',
        'speakers': 'None',
        'ayes': ['Chinchilla', 'Theoharis', 'Antenore', 
            'Hills', 'Joe', 'Martin', 'Mills'],
        'noes': [],
        'vote': '7-0'
    },
    {
        'case_number': '98.238ET',
        'project_name': 'MONTAÑA Bridge District Amendment','
        'project_descr': 'Consideration of a proposal to amend Section 1010 of the City Planning Code to clarify that the Golden Gate Bridge Highway and Transportation District is exempt from the regulations of Article 10 of the Planning Code which authorizes the designation of landmark structures and establishes various procedures for reviewing proposals to demolish or alter landmarks',
        'speakers': ['David Bahlman', 'Gee Gee Platt'],
        'action': 'approved',
        'ayes': ['Chinchilla', 'Theoharis', 'Antenore', 
            'Joe', 'Martin', 'Mills'],
        'excused': ['Hills'],
        'noes': [],
        'vote': '6-0-1',
        'action': 'resolution no. 14633'
    },
    {
        'case_number': '98.341T',
        'project_name': 'MONTAÑA Affordable Child Care Amendment',
        'project_descr': 'Proposal to amend Section 314.5 of the Planning Code to expand the sources and eligible uses of monies in the Affordable Child Care Fund',
        'speakers': [],
        'action': 'approved',
        'ayes': ['Chinchilla', 'Theoharis', 'Antenore', 
            'Hills', 'Joe', 'Martin', 'Mills'],
        'noes': [],
        'vote': '7-0',
        'action': 'resolution no. 14634'
    },
    {
        'case_number': '97,470E',
        'project_address': '475 Brannan Street',
        'lot_number': '31',
        'assessor_block': '3787',
        'project_name': 'Public Hearing on the Draft Environmental Impact Report',
        'project_descr': 'projct would add two stories to an existing two-story-plus-basement building...the project would require a rezoning of the existing 50-foot height limit to 65 feet to permit construction of the 58-foot-tall project',
        'type_district': 'SSO (Service Secondayr Office)',
        'speakers': ['David Bahlman', 'Judy West', 'John Paulson', 'Wendy Earl',''
        'Bob Meyers', 'Jerry Marker', 'Roger Geasham', 'Allison Fuller', 'David Coleen'],
        'action': 'meeting held. public hearing closed',
    },
    {
        'case_number': '98.136C',
        'address': '540 and 590 Van Ness Avenue',
        'lot_number': '13',
        'assessor_block': '766',
        'project_descr': 'conditional use authorization to establish a large fast food restaurant',
        'type_district': 'RC-4',
        'type_district_descr': 'High Density Residential-Commercial',
        'height_and_bulk_district': '130-V',
        'Speakers': ['Robert McCarthy'],
        'action': 'approved with conditions as drafted',
        'ayes': ['Chinchilla', 'Theoharis', 'Antenore', 
            'Hills', 'Joe', 'Martin', 'Mills'],
        'noes': [],
        'vote': '7-0',
        'action': 'resolution no. 14636'
    },
    {
        'case_number': '98.186C',
        'project_address': '4207 Judah Street',
        'lot_number': '37',
        'assessor_block': '1806',
        'project descr': 'conditional use authorization under section 303(e) of planning code to extend termination date of nonconforming bar from 1999 to 2009 to allow bar to continue operating until 2:00 am',
        'type_district': 'RM-1',
        'type_district_descr': 'residential, mixed, low-density',
        'height_and_bulk_district': '40-X',
        'speakers': ['Patrick Than', 'Lil Palermo'],
        'action': 'approved with conditions as drafted',
        'ayes': ['Chinchilla', 'Theoharis', 'Antenore', 
            'Hills', 'Joe', 'Martin', 'Mills'],
        'noes': [],
        'vote': '7-0',
        'action': 'resolution no. 14637'
    },
    {
        'case_number': '98.127C',
        'project_address': '1200 - 9th Avenue',
        'lot_number': '35',
        'lot_number_2': '44',
        'assessor_block': '1742',
        'project_descr': 'modification of previously approved conditional use authorization for development of a lot exceeding 10,000 square feet and establishment of a retial pet store, to allow a general convenience retail/pharmacy use rather than previously approved pet store',
        'type_district': 'NC-2',
        'type_district_descr': 'small-scale neighborhood commercial',
        'height_and_bulk_district': '40-X',
        'speakers': ['Robert McCarthy', 'Steven', 'Lewens',
                     'Bob Planthold', 'Pat Ann Miller', 'Eric Bianco', 'Deboarh Lises',
                     'Dr. Rob Eric', 'Dennis Quinn', 'James Hanley',
                     'Harvey Vickens', 'John Bardis'],
        'action': 'approve with condition as modified',
        'modifications': ['security in parking lot to move cars out within an hour',
                          'pharmacist on site from 8 a.m. to 9 p.m.', 
                          'loading/deliveries on site only', 
                          'van size not to exceed 30 feet', 
                          'hours of operations 7a.m. to 11 p.m.',
                          'project sponsor shall continue to work with staff on project',
                          'sign language as ready by Bob Passmore',
                          'establish a liaison officer', 
                          'loading hours developed in consultation with Planning Department'],
        'ayes': ['chinchilla', 'theoharis', 'martin', 'mills'],
        'noes': ['antenore', 'hills', 'joe'],
        'vote': '4-3',
        'action': 'resolution no. 14638'
    },
    {
        'case_number': '98.285C',
        'project_address': '1640 Stockton Street',
        'lot_number': '15',
        'assessor_block': '103',
        'project_descr': 'request for discretionary review of building permit application no. 9725734, proposing to merge a four-unit building into a single-family dwelling',
        'type_district': 'RH-3',
        'type_district_descr': 'house, three-family',
        'speakers': [],
        'action': 'without hearing, continued to 6/18/98',
        'ayes': ['Chinchilla', 'Theoharis', 'Antenore', 
            'Hills', 'Joe', 'Martin', 'Mills'],
        'noes': [],
        'vote': '7-0',
    },
    {
        'case_number': '98.246D',
        'project_address': '3230 Baker Street',
        'lot_number': '025',
        'assessor_block': '0926',
        'project_descr': 'request for discretionary review of building permit application 9725973, proposing to construct an additional level to existing single-unit house',
        'type_district': 'RH-1',
        'type_district_descr': 'house, one-family',
        'action': 'discretionary review request withdrawn',
    },
    {
        'case_number': '98.254D',
        'project_address': '41 Norfolk Street',
        'lot_number': '051',
        'assessor_block': '3521',
        'project_descr': 'request for '
    }
]