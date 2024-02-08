#include <iostream>
#include <sstream>
#include <time.h>

using namespace std;

class Player {
    public:
    Player(int x, int y){
        health = MAX_HEALTH;
        hunger = MAX_HUNGER;
        thirst = MAX_THIRST;
        
        this->x = x;
        this->y = y;
        this->score = 0;
    }
    
    int getScore() const {
        return score;
    }
    
    void takeDamage(int val){
        this->health -= val;
        if(this->health < 0) this->health = 0;
    }
    
    void takeTurn(){
        this->thirst--;
        this->hunger--;
        
        if(this->thirst <= 0){
            this->health--;
            this->thirst = 0;
        }
        
        if(this->hunger <= 0){
            this->health--;
            this->hunger = 0;
        }
        
        this->score++;
    }

    
    string getStats() const {
        stringstream ss;
        ss  << "============\n"
            << "Health: " << this->health << "\n" 
            << "Hunger: " << this->hunger << "\n" 
            << "Thirst: " << this->thirst << "\n"
            << "============\n";
        return ss.str();
    }
    
    bool isAlive() const {
        return this->health > 0;
    }
    
    void increaseThrist(int val){
        if(this->thirst > MAX_THIRST) this->thirst = MAX_THIRST;
    }
    
    void increaseHunger(int val){
        this->hunger += val;
        if(this->hunger > MAX_HUNGER) this->hunger = MAX_HUNGER;
    }
    
    int x, y;
    private:
    int health, hunger, thirst, score;
    const int MAX_HEALTH = 100; // degbug values
    const int MAX_HUNGER = 100;
    const int MAX_THIRST = 100;
};

class Land {
    public:
    virtual string getDescription() = 0;
    virtual string visit(Player& player) = 0;
};

class Village : public Land {
    public:
    string getDescription(){
        return "a humble village.";
    }
    
    string visit(Player& player){
        int randomNum = rand() % 100;
        
        if(randomNum > 74){
            player.takeDamage(1);
            return "The Villagers chase you off";
        } else {
            player.increaseHunger(1);
            player.increaseThrist(1);
            return "The Villagers welcome you and provide a meal and drink";
        }
    }
};

class RuinedVillage : public Land {
    public:
    string getDescription(){
        return "a ruined village, it has been abandoned for a while";
    }
    
    string visit(Player& player){
        int randomNum = rand() % 100;
        
        if(randomNum > 74){
            player.takeDamage(1);
            return "You feel a sense of unease in the ruined village.";
        } else {
            player.increaseHunger(1);
            player.increaseThrist(1);
            return "You find some old supplies in the ruins.";
        }
    }
};

class BarrenLand : public Land {
    public:
    string getDescription(){
        return "Barren land, nothing grows here";
    }
    
    string visit(Player& player){
        int randomNum = rand() % 100;
        
        if(randomNum > 74){
            player.takeDamage(1);
            return "The desolation of the barren land affects your spirit.";
        } else {
            return "You find nothing of interest in this barren area.";
        }
    }
};

class Forest : public Land {
    public:
    string getDescription(){
        return "a densely wooded forest.";
    }
    
    string visit(Player& player){
        int randomNum = rand() % 100;
        
        if(randomNum > 74){
            player.takeDamage(1);
            return "You are attacked by a bear while foraging for berries.";
        } else {
            player.increaseHunger(1);
            return "You forage for berries in the woods and eat a few.";
        }
    }
};

class Lake : public Land {
    public:
    string getDescription(){
        return "a clear sparkling lake.";
    }
    
    string visit(Player& player){
        player.increaseThrist(2);
        return "You visit the lake and drink its clean water";
    }
};

const int MAP_SIZE = 20;
Land* map[MAP_SIZE][MAP_SIZE];

void populateMap(){
    for(int i = 0; i < MAP_SIZE; i++){
        for(int j = 0; j < MAP_SIZE; j++){
            int randomNum = rand() % 5; // Adjust the range for new land types
            switch(randomNum){
                case 0: // Forest
                    map[i][j] = new Forest;
                    break;
                case 1: // Lake
                    map[i][j] = new Lake;
                    break;
                case 2: // Village
                    map[i][j] = new Village;
                    break;
                case 3: // Ruined Village
                    map[i][j] = new RuinedVillage;
                    break;
                case 4: // Barren Land
                    map[i][j] = new BarrenLand;
                    break;
                default:
                    cout << "Invalid land type selected" << endl;
                    break;
            }
        }
    }
}

int main(){
    srand(time(0));
    
    populateMap();
    
    Player player(MAP_SIZE/2, MAP_SIZE/2);
    
    cout << "You wake up and find yourself lost in the middle of a strange wilderness." << endl;
    
    // TODO: Handle boundary conditions (e.g., if index out of bounds north, you index the south-most location)
    
    while(player.isAlive()){
        cout << "To the north you see " << map[player.x][player.y - 1]->getDescription() << endl;
        cout << "To the east you see " << map[player.x + 1][player.y]->getDescription() << endl;
        cout << "To the south you see " << map[player.x][player.y + 1]->getDescription() << endl;
        cout << "To the west you see " << map[player.x - 1][player.y]->getDescription() << endl;
        
        cout << "Which way will you go? Enter N, E, S, or W:" << endl;
        char userInput;
        cin >> userInput;
        
    switch(userInput){
        case 'N':
            if (player.y > 0) {
                player.y = player.y - 1;
            } else {
                cout << "You can't go further north." << endl;
            }
            break;
        case 'E':
            if (player.x < MAP_SIZE - 1) {
                player.x = player.x + 1;
            } else {
                cout << "You can't go further east." << endl;
            }
            break;
        case 'S':
            if (player.y < MAP_SIZE - 1) {
                player.y = player.y + 1;
            } else {
                cout << "You can't go further south." << endl;
            }
            break;
        case 'W':
            if (player.x > 0) {
                player.x = player.x - 1;
            } else {
                cout << "You can't go further west." << endl;
            }
            break;
        default:
            break;
    }

        cout << map[player.x][player.y]->visit(player) << endl;
        
        cout << player.getStats() << endl;
        player.takeTurn();
    }
    
    cout << "You died." << endl;
    cout << player.getScore() << endl;
    
    return 0;
}