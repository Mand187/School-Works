#include <iostream>
#include <cmath>
#include <vector>
#include <string>


using namespace std;

struct Location {
    string name;
    double latitude;
    double longitude;
};

// Function that returns a vector of Location structures
vector<Location> getLocations() {
    vector<Location> locations = {
        {"Miami Beach, FL, USA", 25.793449, -80.139198}, // 1
        {"Fargo, ND, USA", 46.877186, -96.789803}, // 2 
        {"Idaho City, ID, USA", 43.828850, -115.837860}, // 3
        {"Northampton, MA, USA", 42.328674, -71.217133}, // 4
        {"Newburyport, MA, USA", 42.810356, -70.893875}, // 5
        {"New Bedford, MA, USA", 41.638409, -70.941208}, // 6
        {"Medford, MA, USA", 42.419331, -71.119720}, // 7
        {"Malden, MA, USA", 42.429752, -71.071022}, // 8
        {"Leominster, MA, USA", 42.525482, -71.764183}, // 9
        {"Lawrence, MA, USA", 42.701283, -71.175682} // 10
    };
    return locations;
}

// Function to calculate the distance between two points using the Haversine formula
double harvsineDistance(double lat1, double lon1, double lat2, double lon2) {
    double r = 3958.8;  // radius of the Earth in miles

    // Convert latitude and longitude to radians
    double phi1 = lat1 * M_PI / 180;
    double phi2 = lat2 * M_PI / 180;
    double delta_phi = (lat2 - lat1) * M_PI / 180;
    double delta_lambda = (lon2 - lon1) * M_PI / 180;

    // Apply Haversine formula
    double a = sin(delta_phi / 2) * sin(delta_phi / 2) +
               cos(phi1) * cos(phi2) * sin(delta_lambda / 2) * sin(delta_lambda / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    double d = r * c;

    return d;
}

double getFlightTime(double lat1, double lon1, double lat2, double lon2) {
    
    double s = 500;
    double r = 3958.8;  // radius of the Earth in miles

    // Convert latitude and longitude to radians
    double phi1 = lat1 * M_PI / 180;
    double phi2 = lat2 * M_PI / 180;
    double delta_phi = (lat2 - lat1) * M_PI / 180;
    double delta_lambda = (lon2 - lon1) * M_PI / 180;

    // Apply Haversine formula
    double a = sin(delta_phi / 2) * sin(delta_phi / 2) +
               cos(phi1) * cos(phi2) * sin(delta_lambda / 2) * sin(delta_lambda / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    double d = r * c;

    double t = d / s;

    return t;
}

int main() {
    int userChoice1;
    int userChoice2;
    int vectorNum = 1;
    vector<Location> locations = getLocations();

    for (const Location& location : locations) {
        cout << vectorNum++ << ".";
        cout << " Location: " << location.name;
        cout << " Latitude: " << location.latitude;
        cout << " Longitude: " << location.longitude;
        cout << endl;
    }

    cout << vectorNum++ << ". EXIT *You will have enter twice "<< endl;

    while (true) {
        // Ask User For Destination Choice
        cout << "Enter The Numbers of Your Two Destinations (or enter " << vectorNum - 1 << " to EXIT): ";
        cin >> userChoice1 >> userChoice2;
        cout << endl;

        // Check if the exit option is selected
        if (userChoice1 == vectorNum - 1 || userChoice2 == vectorNum - 1) {
            cout << "Exiting the program..." << endl;
            break;  // End the loop and exit the program
        }
        // Validate user input
        int numLocations = locations.size();
        if (userChoice1 >= 1 && userChoice1 <= numLocations && userChoice2 >= 1 && userChoice2 <= numLocations && userChoice1 != userChoice2) {
            // Get the latitude and longitude of the selected destinations
            string name1 = locations[userChoice1 - 1].name;
            double lat1 = locations[userChoice1 - 1].latitude;
            double lon1 = locations[userChoice1 - 1].longitude;
            string name2 = locations[userChoice2 - 1].name;
            double lat2 = locations[userChoice2 - 1].latitude;
            double lon2 = locations[userChoice2 - 1].longitude;

            // Declare distance variable, then call distance function
            double distance = harvsineDistance(lat1, lon1, lat2, lon2);
            cout << "Distance between " << name1 << " and "<< name2 << " is: " << distance << " miles" << endl;

            // Declare flight time variable, then call flightime function
            double flightTime = getFlightTime(lat1, lon1, lat2, lon2);
            cout << "Flight time between the selected destinations: " << flightTime << " hours" << endl << endl;
        }
        else {
            cout << "Invalid destination choices. Please try again." << endl;
        }
    }

    return 0;
}


