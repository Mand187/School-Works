package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"time"
)

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
	Ingredient string
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
		Title:      "Top Recipe",
		Ingredient: recipe.Title,
	}
	return info, nil
}

func main() {
	// Get API key from environment variable
	apiKey := os.Getenv("SPOONACULAR_API_KEY")
	if apiKey == "" {
		fmt.Println("SPOONACULAR_API_KEY environment variable not set")
		os.Exit(1)
	}

	// Create a client with API key
	client := NewClient(apiKey)

	// Query input
	query := "pasta"

	// Make API call to get recipe
	recipe, err := client.GetRecipe(query)
	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}

	// Print recipe
	fmt.Println("Top Recipe for", recipe.Title, ":")
	fmt.Println("Recipe:", recipe.Ingredient)
}
