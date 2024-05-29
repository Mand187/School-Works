package recipeinfoapi

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type Ingredient struct {
	ID       int     `json:"id"`
	Name     string  `json:"name"`
	Amount   float64 `json:"amount"`
	Unit     string  `json:"unit"`
	Measures struct {
		Metric struct {
			Amount   float64 `json:"amount"`
			UnitLong string  `json:"unitLong"`
			UnitShort string  `json:"unitShort"`
		} `json:"metric"`
		US struct {
			Amount   float64 `json:"amount"`
			UnitLong string  `json:"unitLong"`
			UnitShort string  `json:"unitShort"`
		} `json:"us"`
	} `json:"measures"`
}

type RecipeID int64

type Recipe struct {
	ID          int          `json:"id"`
	Title       string       `json:"title"`
	Image       string       `json:"image"`
	Ingredients []Ingredient `json:"extendedIngredients"`
}

type RecipeInfo struct {
	ID          int
	Title       string
  URL         string
	Ingredients []Ingredient
}

type SpoonRecipeResponse struct {
	ID          int          `json:"id"`
	Title       string       `json:"title"`
  URL         string       `json:"sourceURL"`
	Ingredients []Ingredient `json:"extendedIngredients"`
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

func (c *Client) FormatRecipeInfoURL(recipeID int) string {
	return fmt.Sprintf("%s/recipes/%d/information?includeNutrition=false&apiKey=%s", c.BaseURL, recipeID, c.APIKey)
}

func (c *Client) GetRecipeInfo(recipeID int) (RecipeInfo, error) {
	URL := c.FormatRecipeInfoURL(recipeID)
	resp, err := c.HTTPClient.Get(URL)
	if err != nil {
		return RecipeInfo{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return RecipeInfo{}, fmt.Errorf("could not find recipe with ID: %d", recipeID)
	}
	if resp.StatusCode != http.StatusOK {
		return RecipeInfo{}, fmt.Errorf("unexpected response status %q", resp.Status)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return RecipeInfo{}, err
	}
	recipeInfo, err := parseRecipeInfoResponse(data)
	if err != nil {
		return RecipeInfo{}, err
	}
	return recipeInfo, nil
}

func parseRecipeInfoResponse(data []byte) (RecipeInfo, error) {
	var resp SpoonRecipeResponse
	err := json.Unmarshal(data, &resp)
	if err != nil {
		return RecipeInfo{}, fmt.Errorf("invalid API response %s: %w", data, err)
	}

	ingredients := make([]Ingredient, len(resp.Ingredients))
	for i, ing := range resp.Ingredients {
		ingredient := Ingredient{
			ID:     ing.ID,
			Name:   ing.Name,
			Amount: ing.Amount,
			Unit:   ing.Unit,
		}
		ingredient.Measures.Metric.Amount = ing.Measures.Metric.Amount
		ingredient.Measures.Metric.UnitLong = ing.Measures.Metric.UnitLong
		ingredient.Measures.Metric.UnitShort = ing.Measures.Metric.UnitShort
		ingredient.Measures.US.Amount = ing.Measures.US.Amount
		ingredient.Measures.US.UnitLong = ing.Measures.US.UnitLong
		ingredient.Measures.US.UnitShort = ing.Measures.US.UnitShort

		ingredients[i] = ingredient
	}

	info := RecipeInfo{
		ID:          resp.ID,
		Title:       resp.Title,
    URL:         resp.URL,
		Ingredients: ingredients,
	}
	return info, nil
}
