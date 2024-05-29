package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"io"
	"time"
    "net/url"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const mongoURI = "mongodb://localhost:27017"
const apiKey = "8b5d9ffd5d12429fa3c0edf4c192f9c7"

func handleAboutPage(w http.ResponseWriter, r *http.Request) {
    http.ServeFile(w, r, "FrontEnd/aboutPage.html")
}

func handleContactPage(w http.ResponseWriter, r *http.Request) {
    http.ServeFile(w, r, "FrontEnd/contactPage.html")
}

func handleCookNowPage(w http.ResponseWriter, r *http.Request) {
    http.ServeFile(w, r, "FrontEnd/cookNowPage.html")
}

func handleLandingPage(w http.ResponseWriter, r *http.Request) {
    http.ServeFile(w, r, "FrontEnd/landingPage.html")
}

func handleLoginPage(w http.ResponseWriter, r *http.Request) {
    http.Redirect(w, r, "/pantry", http.StatusSeeOther)
}

func handlePantryPage(w http.ResponseWriter, r *http.Request) {
    http.ServeFile(w, r, "FrontEnd/pantryPage.html")
}

func handleRecipePage(w http.ResponseWriter, r *http.Request) {
    http.ServeFile(w, r, "FrontEnd/recipePage.html")
}

func handleRecipeData(w http.ResponseWriter, r *http.Request) {
    searchQuery := r.URL.Query().Get("query")
    if searchQuery == "" {
        http.Error(w, "Missing search query", http.StatusBadRequest)
        return
    }

    // Create a new Spoonacular API client
    client := NewClient(apiKey)

    // Fetch recipe data from the Spoonacular API using the search query
    recipeInfo, err := client.GetRecipe(searchQuery)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    mongoClient, err := mongo.Connect(context.Background(), options.Client().ApplyURI(mongoURI))
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    defer mongoClient.Disconnect(context.Background())

    recipesCollection := mongoClient.Database("MyPantryDB").Collection("recipes")

    // Insert the recipe data into MongoDB
    _, err = recipesCollection.InsertOne(context.Background(), recipeInfo)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    // Convert the recipe data to JSON format
    recipeJSON, err := json.Marshal(recipeInfo)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    // Set the appropriate headers for the JSON response
    w.Header().Set("Content-Type", "application/json")

    // Write the JSON response
    w.Write(recipeJSON)
}

func handleShoppingListPage(w http.ResponseWriter, r *http.Request) {
    http.ServeFile(w, r, "FrontEnd/shoppingListPage.html")
}

func handleTempMainPage(w http.ResponseWriter, r *http.Request) {
    http.ServeFile(w, r, "FrontEnd/tempMainPage.html")
}

func main() {
    // Initialize router
    router := http.NewServeMux() 

    router.HandleFunc("/about", handleAboutPage)
    router.HandleFunc("/contact", handleContactPage)
    router.HandleFunc("/cook-now", handleCookNowPage)
    router.HandleFunc("/landing", handleLandingPage)
    router.HandleFunc("/login", handleLoginPage)

    router.HandleFunc("/pantry", handlePantryPage)
    router.HandleFunc("/recipes", handleRecipePage)
    router.HandleFunc("/recipe-data", handleRecipeData)

    router.HandleFunc("/shopping-list",handleShoppingListPage)
    router.HandleFunc("/temp-main", handleTempMainPage)
    router.Handle("/", http.FileServer(http.Dir(".")))

    // Start server
    log.Println("Server started on port 127.0.0.1:8000")
    log.Fatal(http.ListenAndServe("127.0.0.1:8000", router))
}


type RecipeID int64
type Recipe struct {
    ID    int    `json:"id"`
    Title string `json:"title"`
    Image string `json:"image"`
}

type RecipeResponse struct {
    Results []Recipe `json:"results"`
}

type RecipeInfo struct {
    ID         int
    Title      string
    // Ingredient string
}

type SpoonRecipeResponse struct {
    Results []struct {
        ID    int    `json:"id"`
        Title string `json:"title"`
    } `json:"results"`
}

type Client struct {
    APIKey     string
    BaseURL    string
    HTTPClient *http.Client
}

func NewClient(key string) *Client {
    return &Client{
        APIKey: key,
        BaseURL: "https://api.spoonacular.com",
        HTTPClient: &http.Client{
            Timeout: 10 * time.Second,
        },
    }
}

func (c *Client) FormatRecipeSearchURL(search string) string {
    search = url.QueryEscape(search)
    return fmt.Sprintf("%s/recipes/complexSearch?query=%s&number=1&apiKey=%s", c.BaseURL, search, c.APIKey)
}

func (c *Client) GetRecipe(search string) (RecipeInfo, error) {
    URL := c.FormatRecipeSearchURL(search)
    resp, err := c.HTTPClient.Get(URL)
    if err != nil {
        return RecipeInfo{}, err
    }
    defer resp.Body.Close()
    if resp.StatusCode == http.StatusNotFound {
        return RecipeInfo{}, fmt.Errorf("could not find search: %s ", search)
    }
    if resp.StatusCode != http.StatusOK {
        return RecipeInfo{}, fmt.Errorf("unexpected response status %q", resp.Status)
    }
    data, err := io.ReadAll(resp.Body)
    if err != nil {
        return RecipeInfo{}, err
    }
    recipe, err := ParseRecipeResponse(data)
    if err != nil {
        return RecipeInfo{}, err
    }
    return recipe, nil
}

func ParseRecipeResponse(data []byte) (RecipeInfo, error) {
    var resp SpoonRecipeResponse
    err := json.Unmarshal(data, &resp)
    if err != nil {
        return RecipeInfo{}, fmt.Errorf("invalid API response %s: %w", data, err)
    }
    if len(resp.Results) == 0 {
        return RecipeInfo{}, fmt.Errorf("no recipes found")
    }
    recipe := resp.Results[0]

    info := RecipeInfo{
        ID:         recipe.ID,
        Title:      recipe.Title,
        // Ingredient: recipe.Title,
    }
    return info, nil
}